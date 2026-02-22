"""MCP tool and resource decorators with observability.

Provides @mcp_tool and @mcp_resource decorators that automatically add
logging, metrics, audit trails, OTel spans, and Prometheus recording.
"""

import functools
import time
from typing import Any, Callable, Optional, TypeVar

from foundry_mcp.core.context import (
    generate_correlation_id,
    get_correlation_id,
    sync_request_context,
)
from foundry_mcp.core.observability.audit import _audit
from foundry_mcp.core.observability.manager import get_observability_manager
from foundry_mcp.core.observability.metrics import _metrics

T = TypeVar("T")


def _record_to_metrics_persistence(
    tool_name: str, success: bool, duration_ms: float, action: Optional[str] = None
) -> None:
    """
    Record tool invocation to metrics persistence for observability history.

    Args:
        tool_name: Name of the tool (router)
        success: Whether the invocation succeeded
        duration_ms: Duration in milliseconds
        action: Optional action name for router tools (e.g., "list", "validate")

    Fails silently if metrics persistence is not configured.
    """
    try:
        from foundry_mcp.core.metrics.persistence import get_metrics_collector

        collector = get_metrics_collector()
        if collector is not None and collector._config.enabled:
            status = "success" if success else "error"
            labels = {"tool": tool_name, "status": status}
            if action:
                labels["action"] = action
            collector.record(
                "tool_invocations_total",
                1.0,
                metric_type="counter",
                labels=labels,
            )
            duration_labels = {"tool": tool_name}
            if action:
                duration_labels["action"] = action
            collector.record(
                "tool_duration_ms",
                duration_ms,
                metric_type="gauge",
                labels=duration_labels,
            )
    except Exception:
        # Never let metrics persistence failures affect tool execution
        pass


def mcp_tool(
    tool_name: Optional[str] = None, emit_metrics: bool = True, audit: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for MCP tool handlers with observability.

    Automatically:
    - Logs tool invocations
    - Emits latency and status metrics
    - Creates audit log entries
    - Creates OTel spans when tracing is enabled
    - Records Prometheus metrics when metrics are enabled

    Args:
        tool_name: Override tool name (defaults to function name)
        emit_metrics: Whether to emit metrics
        audit: Whether to create audit log entries
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = tool_name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            # Set up request context if not already set
            existing_corr_id = get_correlation_id()
            corr_id = existing_corr_id or generate_correlation_id(prefix="tool")

            # Use context manager if we need to establish context
            if not existing_corr_id:
                with sync_request_context(correlation_id=corr_id):
                    return await _async_tool_impl(func, name, corr_id, emit_metrics, audit, *args, **kwargs)
            else:
                return await _async_tool_impl(func, name, corr_id, emit_metrics, audit, *args, **kwargs)

        async def _async_tool_impl(
            func: Callable[..., T],
            name: str,
            corr_id: str,
            emit_metrics: bool,
            audit: bool,
            *args: Any,
            **kwargs: Any,
        ) -> T:
            start = time.perf_counter()
            success = True
            error_msg = None

            # Get observability manager for OTel/Prometheus integration
            manager = get_observability_manager()
            tracer = manager.get_tracer(__name__) if manager.is_tracing_enabled() else None
            prom_exporter = manager.get_prometheus_exporter() if manager.is_metrics_enabled() else None

            # Start Prometheus active operation tracking
            if prom_exporter:
                prom_exporter.record_tool_start(name)

            # Create OTel span if tracing enabled (with correlation_id attribute)
            span_context = None
            if tracer:
                span_context = tracer.start_as_current_span(
                    f"tool:{name}",
                    attributes={
                        "tool.name": name,
                        "tool.type": "mcp_tool",
                        "request.correlation_id": corr_id,
                    },
                )

            try:
                if span_context:
                    with span_context:
                        result = await func(*args, **kwargs)  # type: ignore[misc]
                else:
                    result = await func(*args, **kwargs)  # type: ignore[misc]
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                # Record error in Prometheus
                if prom_exporter:
                    prom_exporter.record_error(name, type(e).__name__)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

                # End Prometheus active operation tracking
                if prom_exporter:
                    prom_exporter.record_tool_end(name)
                    prom_exporter.record_tool_invocation(name, success=success, duration_ms=duration_ms)

                if emit_metrics:
                    labels = {"tool": name, "status": "success" if success else "error"}
                    _metrics.counter("tool.invocations", labels=labels)
                    _metrics.timer("tool.latency", duration_ms, labels={"tool": name})

                if audit:
                    _audit.tool_invocation(
                        tool_name=name,
                        success=success,
                        duration_ms=round(duration_ms, 2),
                        error=error_msg,
                        correlation_id=corr_id,
                    )

                # Record to metrics persistence for observability history
                # Extract action from kwargs for router tools
                action = kwargs.get("action") if isinstance(kwargs.get("action"), str) else None
                _record_to_metrics_persistence(name, success, duration_ms, action=action)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            # Set up request context if not already set
            existing_corr_id = get_correlation_id()
            corr_id = existing_corr_id or generate_correlation_id(prefix="tool")

            # Use context manager if we need to establish context
            if not existing_corr_id:
                with sync_request_context(correlation_id=corr_id):
                    return _sync_tool_impl(func, name, corr_id, emit_metrics, audit, args, kwargs)
            else:
                return _sync_tool_impl(func, name, corr_id, emit_metrics, audit, args, kwargs)

        def _sync_tool_impl(
            _wrapped_func: Callable[..., T],
            _tool_name: str,
            _corr_id: str,
            _emit_metrics: bool,
            _do_audit: bool,
            _args: tuple,
            _kwargs: dict,
        ) -> T:
            """Internal implementation for sync tool execution.

            Note: Parameter names are prefixed with underscore to avoid
            conflicts with tool parameter names (e.g., 'name').
            """
            start = time.perf_counter()
            success = True
            error_msg = None

            # Get observability manager for OTel/Prometheus integration
            manager = get_observability_manager()
            tracer = manager.get_tracer(__name__) if manager.is_tracing_enabled() else None
            prom_exporter = manager.get_prometheus_exporter() if manager.is_metrics_enabled() else None

            # Start Prometheus active operation tracking
            if prom_exporter:
                prom_exporter.record_tool_start(_tool_name)

            # Create OTel span if tracing enabled (with correlation_id attribute)
            span_context = None
            if tracer:
                span_context = tracer.start_as_current_span(
                    f"tool:{_tool_name}",
                    attributes={
                        "tool.name": _tool_name,
                        "tool.type": "mcp_tool",
                        "request.correlation_id": _corr_id,
                    },
                )

            try:
                if span_context:
                    with span_context:
                        result = _wrapped_func(*_args, **_kwargs)
                else:
                    result = _wrapped_func(*_args, **_kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                # Record error in Prometheus
                if prom_exporter:
                    prom_exporter.record_error(_tool_name, type(e).__name__)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

                # End Prometheus active operation tracking
                if prom_exporter:
                    prom_exporter.record_tool_end(_tool_name)
                    prom_exporter.record_tool_invocation(_tool_name, success=success, duration_ms=duration_ms)

                if _emit_metrics:
                    labels = {"tool": _tool_name, "status": "success" if success else "error"}
                    _metrics.counter("tool.invocations", labels=labels)
                    _metrics.timer("tool.latency", duration_ms, labels={"tool": _tool_name})

                if _do_audit:
                    _audit.tool_invocation(
                        tool_name=_tool_name,
                        success=success,
                        duration_ms=round(duration_ms, 2),
                        error=error_msg,
                        correlation_id=_corr_id,
                    )

                # Record to metrics persistence for observability history
                # Extract action from kwargs for router tools
                action = _kwargs.get("action") if isinstance(_kwargs.get("action"), str) else None
                _record_to_metrics_persistence(_tool_name, success, duration_ms, action=action)

        # Return appropriate wrapper based on whether func is async
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[reportReturnType]
        return sync_wrapper

    return decorator


def mcp_resource(
    resource_type: Optional[str] = None, emit_metrics: bool = True, audit: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for MCP resource handlers with observability.

    Automatically:
    - Logs resource access
    - Emits latency and status metrics
    - Creates audit log entries
    - Creates OTel spans when tracing is enabled
    - Records Prometheus metrics when metrics are enabled

    Args:
        resource_type: Type of resource (e.g., "spec", "journal")
        emit_metrics: Whether to emit metrics
        audit: Whether to create audit log entries
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        rtype = resource_type or "resource"

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            success = True
            error_msg = None
            resource_id = kwargs.get("spec_id") or kwargs.get("id") or "unknown"

            # Get observability manager for OTel/Prometheus integration
            manager = get_observability_manager()
            tracer = manager.get_tracer(__name__) if manager.is_tracing_enabled() else None
            prom_exporter = manager.get_prometheus_exporter() if manager.is_metrics_enabled() else None

            # Create OTel span if tracing enabled
            span_context = None
            if tracer:
                span_context = tracer.start_as_current_span(
                    f"resource:{rtype}",
                    attributes={
                        "resource.type": rtype,
                        "resource.id": str(resource_id),
                    },
                )

            try:
                if span_context:
                    with span_context:
                        result = await func(*args, **kwargs)  # type: ignore[misc]
                else:
                    result = await func(*args, **kwargs)  # type: ignore[misc]
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                # Record error in Prometheus
                if prom_exporter:
                    prom_exporter.record_error(f"resource:{rtype}", type(e).__name__)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

                # Record Prometheus resource access
                if prom_exporter:
                    prom_exporter.record_resource_access(rtype, "read")

                if emit_metrics:
                    labels = {
                        "resource_type": rtype,
                        "status": "success" if success else "error",
                    }
                    _metrics.counter("resource.access", labels=labels)
                    _metrics.timer("resource.latency", duration_ms, labels={"resource_type": rtype})

                if audit:
                    _audit.resource_access(
                        resource_type=rtype,
                        resource_id=str(resource_id),
                        action="read",
                        success=success,
                        duration_ms=round(duration_ms, 2),
                        error=error_msg,
                    )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            success = True
            error_msg = None
            resource_id = kwargs.get("spec_id") or kwargs.get("id") or "unknown"

            # Get observability manager for OTel/Prometheus integration
            manager = get_observability_manager()
            tracer = manager.get_tracer(__name__) if manager.is_tracing_enabled() else None
            prom_exporter = manager.get_prometheus_exporter() if manager.is_metrics_enabled() else None

            # Create OTel span if tracing enabled
            span_context = None
            if tracer:
                span_context = tracer.start_as_current_span(
                    f"resource:{rtype}",
                    attributes={
                        "resource.type": rtype,
                        "resource.id": str(resource_id),
                    },
                )

            try:
                if span_context:
                    with span_context:
                        result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                # Record error in Prometheus
                if prom_exporter:
                    prom_exporter.record_error(f"resource:{rtype}", type(e).__name__)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

                # Record Prometheus resource access
                if prom_exporter:
                    prom_exporter.record_resource_access(rtype, "read")

                if emit_metrics:
                    labels = {
                        "resource_type": rtype,
                        "status": "success" if success else "error",
                    }
                    _metrics.counter("resource.access", labels=labels)
                    _metrics.timer("resource.latency", duration_ms, labels={"resource_type": rtype})

                if audit:
                    _audit.resource_access(
                        resource_type=rtype,
                        resource_id=str(resource_id),
                        action="read",
                        success=success,
                        duration_ms=round(duration_ms, 2),
                        error=error_msg,
                    )

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[reportReturnType]
        return sync_wrapper

    return decorator
