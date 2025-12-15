"""Tests for observability graceful degradation.

Verifies that observability features work correctly when:
- Dependencies are not installed (no-op behavior)
- Observability is disabled in config
- Only partial dependencies are available
"""

import pytest


class TestNoOpStubs:
    """Test no-op stub implementations."""

    def test_noop_tracer_can_be_imported(self):
        """NoOp tracer should always be importable."""
        from foundry_mcp.core.otel_stubs import get_noop_tracer

        tracer = get_noop_tracer("test")
        assert tracer is not None

    def test_noop_tracer_context_manager(self):
        """NoOp tracer spans should work as context managers."""
        from foundry_mcp.core.otel_stubs import get_noop_tracer

        tracer = get_noop_tracer("test")
        with tracer.start_as_current_span("test-span") as span:
            assert span is not None
            span.set_attribute("key", "value")  # Should not raise

    def test_noop_meter_can_be_imported(self):
        """NoOp meter should always be importable."""
        from foundry_mcp.core.otel_stubs import get_noop_meter

        meter = get_noop_meter("test")
        assert meter is not None

    def test_noop_meter_instruments(self):
        """NoOp meter instruments should be callable."""
        from foundry_mcp.core.otel_stubs import get_noop_meter

        meter = get_noop_meter("test")
        counter = meter.create_counter("test_counter")
        counter.add(1, {"label": "value"})  # Should not raise

        gauge = meter.create_gauge("test_gauge")
        gauge.set(42, {"label": "value"})  # Should not raise

        histogram = meter.create_histogram("test_histogram")
        histogram.record(0.5, {"label": "value"})  # Should not raise


class TestObservabilityStatus:
    """Test get_observability_status() function."""

    def test_status_returns_dict(self):
        """Status should return a dict with expected keys."""
        from foundry_mcp.core.observability import get_observability_status

        status = get_observability_status()
        assert isinstance(status, dict)
        assert "opentelemetry_available" in status
        assert "prometheus_available" in status
        assert "opentelemetry_enabled" in status
        assert "version" in status

    def test_status_availability_flags_are_bool(self):
        """Availability flags should be booleans."""
        from foundry_mcp.core.observability import get_observability_status

        status = get_observability_status()
        assert isinstance(status["opentelemetry_available"], bool)
        assert isinstance(status["prometheus_available"], bool)
        assert isinstance(status["opentelemetry_enabled"], bool)


class TestObservabilityManager:
    """Test ObservabilityManager with disabled config."""

    def test_manager_singleton(self):
        """Manager should be a singleton."""
        from foundry_mcp.core.observability import get_observability_manager

        manager1 = get_observability_manager()
        manager2 = get_observability_manager()
        assert manager1 is manager2

    def test_manager_disabled_by_default(self):
        """Manager should have tracing/metrics disabled by default."""
        from foundry_mcp.core.observability import ObservabilityManager

        # Create a fresh manager instance for testing
        manager = ObservabilityManager.__new__(ObservabilityManager)
        manager._initialized = False
        manager._config = None
        manager._otel_initialized = False
        manager._prometheus_initialized = False

        assert manager.is_tracing_enabled() is False
        assert manager.is_metrics_enabled() is False

    def test_manager_returns_noop_tracer_when_disabled(self):
        """Manager should return no-op tracer when tracing is disabled."""
        from foundry_mcp.core.observability import ObservabilityManager

        manager = ObservabilityManager.__new__(ObservabilityManager)
        manager._initialized = False
        manager._config = None
        manager._otel_initialized = False
        manager._prometheus_initialized = False

        tracer = manager.get_tracer("test")
        assert tracer is not None

        # Should work as context manager without errors
        with tracer.start_as_current_span("test"):
            pass

    def test_manager_returns_noop_exporter_when_disabled(self):
        """Manager should return no-op exporter when metrics are disabled."""
        from foundry_mcp.core.observability import ObservabilityManager

        manager = ObservabilityManager.__new__(ObservabilityManager)
        manager._initialized = False
        manager._config = None
        manager._otel_initialized = False
        manager._prometheus_initialized = False

        exporter = manager.get_prometheus_exporter()
        assert exporter is not None

        # All methods should be callable without errors
        exporter.record_tool_invocation("test", success=True, duration_ms=1.0)
        exporter.record_tool_start("test")
        exporter.record_tool_end("test")
        exporter.record_resource_access("spec", "read")
        exporter.record_error("test", "TestError")


class TestMcpToolDecoratorFastPath:
    """Test @mcp_tool decorator behavior when observability is disabled."""

    def test_decorator_works_without_observability(self):
        """Decorator should work correctly when observability is disabled."""
        from foundry_mcp.core.observability import mcp_tool

        @mcp_tool(tool_name="test_tool")
        def my_tool(x: int) -> int:
            return x * 2

        result = my_tool(21)
        assert result == 42

    def test_async_decorator_works_without_observability(self):
        """Async decorator should work correctly when observability is disabled."""
        import asyncio
        from foundry_mcp.core.observability import mcp_tool

        @mcp_tool(tool_name="test_async_tool")
        async def my_async_tool(x: int) -> int:
            return x * 2

        result = asyncio.run(my_async_tool(21))
        assert result == 42

    def test_decorator_handles_exceptions(self):
        """Decorator should propagate exceptions correctly."""
        from foundry_mcp.core.observability import mcp_tool

        @mcp_tool(tool_name="failing_tool")
        def failing_tool():
            raise ValueError("Expected error")

        with pytest.raises(ValueError, match="Expected error"):
            failing_tool()


class TestConfigLoading:
    """Test configuration loading for observability."""

    def test_observability_config_defaults(self):
        """ObservabilityConfig should have sensible defaults."""
        from foundry_mcp.config import ObservabilityConfig

        config = ObservabilityConfig()
        assert config.enabled is False
        assert config.otel_enabled is False
        assert config.otel_endpoint == "localhost:4317"
        assert config.otel_service_name == "foundry-mcp"
        assert config.otel_sample_rate == 1.0
        assert config.prometheus_enabled is False
        assert config.prometheus_port == 0
        assert config.prometheus_host == "0.0.0.0"
        assert config.prometheus_namespace == "foundry_mcp"

    def test_observability_config_from_dict(self):
        """ObservabilityConfig should load from TOML dict."""
        from foundry_mcp.config import ObservabilityConfig

        data = {
            "enabled": True,
            "otel_enabled": True,
            "otel_endpoint": "otel:4317",
            "prometheus_enabled": True,
            "prometheus_port": 9090,
        }
        config = ObservabilityConfig.from_toml_dict(data)

        assert config.enabled is True
        assert config.otel_enabled is True
        assert config.otel_endpoint == "otel:4317"
        assert config.prometheus_enabled is True
        assert config.prometheus_port == 9090

    def test_server_config_has_observability(self):
        """ServerConfig should include observability settings."""
        from foundry_mcp.config import ServerConfig

        config = ServerConfig()
        assert hasattr(config, "observability")
        assert config.observability.enabled is False


class TestMetricsCollector:
    """Test MetricsCollector behavior."""

    def test_metrics_collector_works_without_prometheus(self):
        """MetricsCollector should work even when Prometheus is disabled."""
        from foundry_mcp.core.observability import MetricsCollector

        collector = MetricsCollector(prefix="test")

        # All methods should work without raising
        collector.counter("test_counter", 1, {"label": "value"})
        collector.gauge("test_gauge", 42, {"label": "value"})
        collector.timer("test_timer", 100.5, {"label": "value"})
        collector.histogram("test_histogram", 0.5, {"label": "value"})
