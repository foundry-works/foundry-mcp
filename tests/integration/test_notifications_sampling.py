"""
Integration tests for MCP Notifications, Sampling, and Rate Limiting.

Tests:
- Notification emission on spec/resource changes
- Sampling request handling and impact analysis
- Rate limit middleware and throttling events
"""

import json
import os
import pytest
import time
from unittest.mock import Mock, patch
from foundry_mcp.server import create_server
from foundry_mcp.config import ServerConfig
from foundry_mcp.core.capabilities import (
    Notification,
    NotificationManager,
    SamplingRequest,
    SamplingResponse,
    SamplingManager,
    CapabilitiesRegistry,
    get_capabilities_registry,
    get_notification_manager,
    get_sampling_manager,
)
from foundry_mcp.core.rate_limit import (
    RateLimitConfig,
    RateLimitResult,
    TokenBucketLimiter,
    RateLimitManager,
    get_rate_limit_manager,
    check_rate_limit,
)


@pytest.fixture
def test_specs_dir(tmp_path):
    """Create a test specs directory with sample spec."""
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "active").mkdir()
    (specs_dir / "pending").mkdir()
    (specs_dir / "completed").mkdir()
    (specs_dir / "archived").mkdir()

    # Create a sample spec
    sample_spec = {
        "spec_id": "test-spec-001",
        "title": "Test Specification",
        "metadata": {
            "title": "Test Specification",
            "description": "A test spec",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Specification",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Implementation",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1"],
            },
            "task-1-1": {
                "type": "task",
                "title": "First task",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
            },
        },
        "journal": [],
    }

    spec_file = specs_dir / "active" / "test-spec-001.json"
    with open(spec_file, "w") as f:
        json.dump(sample_spec, f)

    return specs_dir


@pytest.fixture
def test_config(test_specs_dir):
    """Create a test server configuration."""
    return ServerConfig(
        server_name="foundry-mcp-test",
        server_version="0.1.0",
        specs_dir=test_specs_dir,
        log_level="WARNING",
    )


@pytest.fixture
def notification_manager():
    """Create a fresh NotificationManager for testing."""
    return NotificationManager()


@pytest.fixture
def sampling_manager():
    """Create a fresh SamplingManager for testing."""
    return SamplingManager()


@pytest.fixture
def rate_limit_manager():
    """Create a fresh RateLimitManager for testing."""
    return RateLimitManager()


# ============================================================================
# Notification Tests
# ============================================================================


class TestNotificationEmission:
    """Tests for notification emission on resource changes."""

    def test_notification_created_with_timestamp(self):
        """Test that notifications are created with timestamp."""
        notification = Notification(method="test/method", params={"key": "value"})
        assert notification.method == "test/method"
        assert notification.params == {"key": "value"}
        assert notification.timestamp != ""
        assert "Z" in notification.timestamp  # UTC format

    def test_notification_manager_emit(self, notification_manager):
        """Test that notifications are emitted to handlers."""
        received = []
        def handler(n):
            return received.append(n)
        notification_manager.register_handler(handler)

        notification = Notification(method="test/method")
        notification_manager.emit(notification)

        assert len(received) == 1
        assert received[0].method == "test/method"

    def test_notification_manager_multiple_handlers(self, notification_manager):
        """Test that notifications are sent to all handlers."""
        received_1 = []
        received_2 = []
        notification_manager.register_handler(lambda n: received_1.append(n))
        notification_manager.register_handler(lambda n: received_2.append(n))

        notification_manager.emit(Notification(method="test/method"))

        assert len(received_1) == 1
        assert len(received_2) == 1

    def test_notification_manager_unregister(self, notification_manager):
        """Test handler unregistration."""
        received = []
        def handler(n):
            return received.append(n)
        notification_manager.register_handler(handler)
        notification_manager.unregister_handler(handler)

        notification_manager.emit(Notification(method="test/method"))

        assert len(received) == 0

    def test_emit_resource_update(self, notification_manager):
        """Test resource update notification emission."""
        received = []
        notification_manager.register_handler(lambda n: received.append(n))

        notification_manager.emit_resource_update(
            uri="foundry://specs/test-spec-001",
            update_type="updated",
            metadata={"field": "status"}
        )

        assert len(received) == 1
        assert received[0].method == "notifications/resources/updated"
        assert received[0].params["uri"] == "foundry://specs/test-spec-001"
        assert received[0].params["type"] == "updated"

    def test_emit_spec_updated(self, notification_manager):
        """Test spec update notification emission."""
        received = []
        notification_manager.register_handler(lambda n: received.append(n))

        notification_manager.emit_spec_updated(
            spec_id="test-spec-001",
            update_type="status_changed",
            changes={"old_status": "pending", "new_status": "in_progress"}
        )

        assert len(received) == 1
        assert received[0].method == "foundry/specs/updated"
        assert received[0].params["spec_id"] == "test-spec-001"
        assert received[0].params["type"] == "status_changed"

    def test_emit_task_updated(self, notification_manager):
        """Test task update notification emission."""
        received = []
        notification_manager.register_handler(lambda n: received.append(n))

        notification_manager.emit_task_updated(
            spec_id="test-spec-001",
            task_id="task-1-1",
            update_type="status_changed",
            changes={"old_status": "pending", "new_status": "completed"}
        )

        assert len(received) == 1
        assert received[0].method == "foundry/tasks/updated"
        assert received[0].params["spec_id"] == "test-spec-001"
        assert received[0].params["task_id"] == "task-1-1"

    def test_notification_pause_resume(self, notification_manager):
        """Test notification pausing and resuming."""
        received = []
        notification_manager.register_handler(lambda n: received.append(n))

        notification_manager.pause()
        notification_manager.emit(Notification(method="paused/1"))
        notification_manager.emit(Notification(method="paused/2"))

        # Nothing received while paused
        assert len(received) == 0

        notification_manager.resume()

        # Pending notifications flushed on resume
        assert len(received) == 2
        assert received[0].method == "paused/1"
        assert received[1].method == "paused/2"

    def test_clear_pending_notifications(self, notification_manager):
        """Test clearing pending notifications."""
        received = []
        notification_manager.register_handler(lambda n: received.append(n))

        notification_manager.pause()
        notification_manager.emit(Notification(method="pending/1"))
        notification_manager.clear_pending()
        notification_manager.resume()

        assert len(received) == 0

    def test_handler_exception_does_not_stop_others(self, notification_manager):
        """Test that handler exceptions don't prevent other handlers."""
        received = []

        def failing_handler(n):
            raise Exception("Handler error")

        notification_manager.register_handler(failing_handler)
        notification_manager.register_handler(lambda n: received.append(n))

        notification_manager.emit(Notification(method="test/method"))

        # Second handler still receives notification
        assert len(received) == 1


# ============================================================================
# Sampling Tests
# ============================================================================


class TestSamplingRequests:
    """Tests for MCP Sampling request handling."""

    def test_sampling_unavailable_by_default(self, sampling_manager):
        """Test that sampling is unavailable without handler."""
        assert not sampling_manager.is_available()

    def test_sampling_available_with_handler(self, sampling_manager):
        """Test that sampling is available after setting handler."""
        mock_handler = Mock(return_value=SamplingResponse(content="test", model="test"))
        sampling_manager.set_handler(mock_handler)
        assert sampling_manager.is_available()

    def test_sampling_request_without_handler(self, sampling_manager):
        """Test that sampling request returns None without handler."""
        request = SamplingRequest(messages=[{"role": "user", "content": "test"}])
        result = sampling_manager.request(request)
        assert result is None

    def test_sampling_request_with_handler(self, sampling_manager):
        """Test successful sampling request."""
        mock_response = SamplingResponse(
            content="Test response",
            model="claude-3",
            usage={"total_tokens": 100}
        )
        mock_handler = Mock(return_value=mock_response)
        sampling_manager.set_handler(mock_handler)

        request = SamplingRequest(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=500
        )
        result = sampling_manager.request(request)

        assert result is not None
        assert result.content == "Test response"
        mock_handler.assert_called_once_with(request)

    def test_sampling_stats_tracking(self, sampling_manager):
        """Test that sampling tracks statistics."""
        mock_response = SamplingResponse(
            content="test",
            model="claude-3",
            usage={"total_tokens": 50}
        )
        sampling_manager.set_handler(Mock(return_value=mock_response))

        request = SamplingRequest(messages=[{"role": "user", "content": "test"}])
        sampling_manager.request(request)
        sampling_manager.request(request)

        stats = sampling_manager.get_stats()
        assert stats["request_count"] == 2
        assert stats["total_tokens"] == 100

    def test_analyze_impact_unavailable(self, sampling_manager):
        """Test impact analysis when sampling unavailable."""
        result = sampling_manager.analyze_impact("TestClass", "class")
        assert result is None

    def test_analyze_impact_with_handler(self, sampling_manager):
        """Test impact analysis with sampling handler."""
        analysis_result = {
            "direct_impacts": ["ClassA", "ClassB"],
            "indirect_impacts": ["ClassC"],
            "risk_level": "medium",
            "recommendations": ["Review tests"]
        }
        mock_response = SamplingResponse(
            content=json.dumps(analysis_result),
            model="claude-3",
            usage={"total_tokens": 100}
        )
        sampling_manager.set_handler(Mock(return_value=mock_response))

        result = sampling_manager.analyze_impact(
            target="TestClass",
            target_type="class",
            context="Changing method signature"
        )

        assert result is not None
        assert result["risk_level"] == "medium"
        assert "ClassA" in result["direct_impacts"]

    def test_analyze_impact_invalid_json_response(self, sampling_manager):
        """Test impact analysis with invalid JSON response."""
        mock_response = SamplingResponse(
            content="This is not JSON",
            model="claude-3"
        )
        sampling_manager.set_handler(Mock(return_value=mock_response))

        result = sampling_manager.analyze_impact("TestClass", "class")

        assert result is not None
        assert "raw_response" in result

    def test_sampling_handler_exception(self, sampling_manager):
        """Test handling of sampling handler exceptions."""
        def failing_handler(request):
            raise Exception("Handler error")

        sampling_manager.set_handler(failing_handler)
        request = SamplingRequest(messages=[{"role": "user", "content": "test"}])

        result = sampling_manager.request(request)
        assert result is None


# ============================================================================
# Rate Limit Tests
# ============================================================================


class TestTokenBucketLimiter:
    """Tests for token bucket rate limiter."""

    def test_limiter_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        config = RateLimitConfig(requests_per_minute=60, burst_limit=5)
        limiter = TokenBucketLimiter(config)

        for i in range(5):
            result = limiter.acquire()
            assert result.allowed, f"Request {i+1} should be allowed"

    def test_limiter_blocks_over_burst(self):
        """Test that requests over burst limit are blocked."""
        config = RateLimitConfig(requests_per_minute=60, burst_limit=3)
        limiter = TokenBucketLimiter(config)

        # Exhaust burst
        for _ in range(3):
            limiter.acquire()

        # Next request should be blocked
        result = limiter.acquire()
        assert not result.allowed
        assert "Rate limit" in result.reason or result.reason == ""

    def test_limiter_refills_over_time(self):
        """Test that tokens refill over time."""
        config = RateLimitConfig(requests_per_minute=60, burst_limit=2)
        limiter = TokenBucketLimiter(config)

        # Exhaust tokens
        limiter.acquire()
        limiter.acquire()
        result = limiter.acquire()
        assert not result.allowed

        # Wait for refill (1 second at 60 RPM = 1 token)
        time.sleep(1.1)
        result = limiter.acquire()
        assert result.allowed

    def test_limiter_check_without_consuming(self):
        """Test check method doesn't consume tokens."""
        config = RateLimitConfig(requests_per_minute=60, burst_limit=1)
        limiter = TokenBucketLimiter(config)

        # Check shouldn't consume
        for _ in range(5):
            result = limiter.check()
            assert result.allowed

        # Acquire should still work (first token available)
        result = limiter.acquire()
        assert result.allowed

    def test_limiter_disabled(self):
        """Test that disabled limiter always allows."""
        config = RateLimitConfig(requests_per_minute=60, burst_limit=1, enabled=False)
        limiter = TokenBucketLimiter(config)

        for _ in range(100):
            result = limiter.acquire()
            assert result.allowed
            assert result.remaining == -1  # Indicates disabled

    def test_limiter_stats(self):
        """Test limiter statistics."""
        config = RateLimitConfig(requests_per_minute=10, burst_limit=3)
        limiter = TokenBucketLimiter(config)

        limiter.acquire()
        limiter.acquire()
        limiter.acquire()
        limiter.acquire()  # This should be throttled

        stats = limiter.get_stats()
        assert stats["requests"] == 4
        assert stats["throttled"] == 1
        assert stats["limit"] == 10
        assert stats["burst"] == 3


class TestRateLimitManager:
    """Tests for rate limit manager."""

    def test_manager_creates_limiters(self, rate_limit_manager):
        """Test that manager creates limiters on demand."""
        limiter = rate_limit_manager.get_limiter("test_tool")
        assert limiter is not None
        assert isinstance(limiter, TokenBucketLimiter)

    def test_manager_reuses_limiters(self, rate_limit_manager):
        """Test that manager reuses existing limiters."""
        limiter1 = rate_limit_manager.get_limiter("test_tool")
        limiter2 = rate_limit_manager.get_limiter("test_tool")
        assert limiter1 is limiter2

    def test_manager_per_tenant_limiters(self, rate_limit_manager):
        """Test per-tenant limiter isolation."""
        limiter_a = rate_limit_manager.get_limiter("test_tool", tenant_id="tenant_a")
        limiter_b = rate_limit_manager.get_limiter("test_tool", tenant_id="tenant_b")
        assert limiter_a is not limiter_b

    def test_check_limit_enforces(self, rate_limit_manager):
        """Test that check_limit enforces rate limits."""
        # Configure a strict limit
        rate_limit_manager._tool_configs["strict_tool"] = RateLimitConfig(
            requests_per_minute=60,
            burst_limit=2
        )

        # First two should pass
        assert rate_limit_manager.check_limit("strict_tool", log_on_throttle=False).allowed
        assert rate_limit_manager.check_limit("strict_tool", log_on_throttle=False).allowed

        # Third should fail
        result = rate_limit_manager.check_limit("strict_tool", log_on_throttle=False)
        assert not result.allowed

    def test_manager_stats(self, rate_limit_manager):
        """Test manager statistics collection."""
        rate_limit_manager.check_limit("tool1", log_on_throttle=False)
        rate_limit_manager.check_limit("tool2", log_on_throttle=False)

        stats = rate_limit_manager.get_all_stats()
        assert "tools" in stats
        assert "tool1" in stats["tools"]
        assert "tool2" in stats["tools"]
        assert "schema_version" in stats

    def test_manager_reset_specific_tool(self, rate_limit_manager):
        """Test resetting specific tool limiter."""
        rate_limit_manager.check_limit("tool_to_reset", log_on_throttle=False)
        assert "tool_to_reset" in rate_limit_manager._limiters

        rate_limit_manager.reset(tool_name="tool_to_reset")
        assert "tool_to_reset" not in rate_limit_manager._limiters

    def test_manager_reset_all(self, rate_limit_manager):
        """Test resetting all limiters."""
        rate_limit_manager.check_limit("tool1", log_on_throttle=False)
        rate_limit_manager.check_limit("tool2", log_on_throttle=False)

        rate_limit_manager.reset()

        assert len(rate_limit_manager._limiters) == 0

    def test_load_from_env(self, rate_limit_manager):
        """Test loading config from environment variables."""
        with patch.dict(os.environ, {
            "FOUNDRY_RATE_LIMIT_DEFAULT": "30",
            "FOUNDRY_RATE_LIMIT_BURST": "5"
        }):
            rate_limit_manager.load_from_env()

        assert rate_limit_manager._global_config.requests_per_minute == 30
        assert rate_limit_manager._global_config.burst_limit == 5


class TestRateLimitThrottleEvents:
    """Tests for rate limit throttle event handling."""

    def test_throttle_logs_audit(self, rate_limit_manager):
        """Test that throttle events are logged."""
        rate_limit_manager._tool_configs["throttle_test"] = RateLimitConfig(
            requests_per_minute=60,
            burst_limit=1,
            reason="Test rate limit"
        )

        # Exhaust limit
        rate_limit_manager.check_limit("throttle_test", log_on_throttle=False)

        # This should trigger throttle
        with patch("foundry_mcp.core.rate_limit.audit_log") as mock_audit:
            result = rate_limit_manager.check_limit("throttle_test", log_on_throttle=True)
            assert not result.allowed
            mock_audit.assert_called_once()
            call_args = mock_audit.call_args
            assert call_args[0][0] == "rate_limit_exceeded"

    def test_auth_failure_logging(self, rate_limit_manager):
        """Test authentication failure logging."""
        with patch("foundry_mcp.core.rate_limit.audit_log") as mock_audit:
            rate_limit_manager.log_auth_failure(
                tool_name="test_tool",
                tenant_id="test_tenant",
                reason="Invalid API key"
            )

            mock_audit.assert_called_once()
            call_args = mock_audit.call_args
            assert call_args[0][0] == "auth_failure"
            assert call_args[1]["reason"] == "Invalid API key"


# ============================================================================
# Capabilities Registry Tests
# ============================================================================


class TestCapabilitiesRegistry:
    """Tests for capabilities registry."""

    def test_registry_has_notification_manager(self):
        """Test that registry provides notification manager."""
        registry = CapabilitiesRegistry()
        assert registry.notifications is not None
        assert isinstance(registry.notifications, NotificationManager)

    def test_registry_has_sampling_manager(self):
        """Test that registry provides sampling manager."""
        registry = CapabilitiesRegistry()
        assert registry.sampling is not None
        assert isinstance(registry.sampling, SamplingManager)

    def test_registry_default_capabilities(self):
        """Test default capability settings."""
        registry = CapabilitiesRegistry()
        assert registry.is_enabled("notifications")
        assert not registry.is_enabled("sampling")  # Requires handler
        assert registry.is_enabled("resources")
        assert registry.is_enabled("prompts")
        assert registry.is_enabled("tools")

    def test_registry_enable_disable(self):
        """Test enabling/disabling capabilities."""
        registry = CapabilitiesRegistry()

        registry.disable("notifications")
        assert not registry.is_enabled("notifications")

        registry.enable("notifications")
        assert registry.is_enabled("notifications")

    def test_registry_get_capabilities(self):
        """Test get_capabilities returns full info."""
        registry = CapabilitiesRegistry()
        caps = registry.get_capabilities()

        assert "schema_version" in caps
        assert "capabilities" in caps
        assert "notifications" in caps
        assert "sampling" in caps

    def test_registry_metadata(self):
        """Test capability metadata storage."""
        registry = CapabilitiesRegistry()
        registry.set_metadata("version", "1.0.0")
        registry.set_metadata("author", "test")

        caps = registry.get_capabilities()
        assert caps["metadata"]["version"] == "1.0.0"
        assert caps["metadata"]["author"] == "test"


# ============================================================================
# Integration Tests
# ============================================================================


class TestNotificationIntegration:
    """Integration tests for notifications with server."""

    def test_global_notification_manager(self):
        """Test global notification manager accessor."""
        manager = get_notification_manager()
        assert manager is not None
        assert isinstance(manager, NotificationManager)

    def test_global_sampling_manager(self):
        """Test global sampling manager accessor."""
        manager = get_sampling_manager()
        assert manager is not None
        assert isinstance(manager, SamplingManager)

    def test_global_registry(self):
        """Test global capabilities registry."""
        registry = get_capabilities_registry()
        assert registry is not None
        assert isinstance(registry, CapabilitiesRegistry)


class TestRateLimitIntegration:
    """Integration tests for rate limiting."""

    def test_global_rate_limit_manager(self):
        """Test global rate limit manager accessor."""
        manager = get_rate_limit_manager()
        assert manager is not None
        assert isinstance(manager, RateLimitManager)

    def test_check_rate_limit_helper(self):
        """Test check_rate_limit helper function."""
        result = check_rate_limit("test_integration_tool")
        assert isinstance(result, RateLimitResult)
        assert result.allowed  # First request should pass


class TestServerCapabilitiesIntegration:
    """Integration tests for server capabilities."""

    def test_server_creates_with_capabilities(self, test_config):
        """Test that server initializes capabilities."""
        server = create_server(test_config)
        assert server is not None

    def test_notification_on_spec_operations(self, test_config):
        """Test notifications during spec operations."""
        manager = get_notification_manager()
        received = []
        manager.register_handler(lambda n: received.append(n))

        # Emit a notification manually (as tools would)
        manager.emit_spec_updated("test-spec-001", "updated")

        # Should have received notification
        assert len(received) >= 1

        # Clean up
        manager.unregister_handler(received.append)
