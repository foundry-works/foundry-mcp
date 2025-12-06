# Observability Guide

This guide covers foundry-mcp's observability stack, including OpenTelemetry tracing and Prometheus metrics.

## Design Philosophy

foundry-mcp implements **graceful degradation** for all observability features:

- When optional dependencies are not installed, all observability features become no-ops
- Zero performance overhead when observability is disabled
- No import errors or runtime failures from missing dependencies
- Features can be enabled incrementally as needed

This means you can:
1. Run foundry-mcp without any observability dependencies
2. Add Prometheus metrics without OpenTelemetry
3. Add OpenTelemetry tracing without Prometheus
4. Enable full observability with both

## Quick Start

### Installing Dependencies

```bash
# Prometheus metrics only
pip install foundry-mcp[metrics]

# OpenTelemetry tracing only
pip install foundry-mcp[tracing]

# Full observability stack
pip install foundry-mcp[observability]
```

### Enabling Prometheus Metrics

Add to your `foundry-mcp.toml`:

```toml
[observability]
enabled = true
prometheus_enabled = true
prometheus_port = 9090  # Expose /metrics on this port
```

Verify metrics are available:

```bash
curl http://localhost:9090/metrics
```

### Enabling OpenTelemetry Tracing

Add to your `foundry-mcp.toml`:

```toml
[observability]
enabled = true
otel_enabled = true
otel_endpoint = "localhost:4317"  # OTLP gRPC endpoint
otel_service_name = "foundry-mcp"
otel_sample_rate = 1.0  # Sample all traces (reduce in production)
```

### Full Stack Configuration

```toml
[observability]
enabled = true

# OpenTelemetry
otel_enabled = true
otel_endpoint = "otel-collector:4317"
otel_service_name = "foundry-mcp"
otel_sample_rate = 0.1  # Sample 10% in production

# Prometheus
prometheus_enabled = true
prometheus_port = 9090
prometheus_host = "0.0.0.0"
prometheus_namespace = "foundry_mcp"
```

## Configuration Reference

### TOML Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `observability.enabled` | bool | `false` | Master switch for all observability |
| `observability.otel_enabled` | bool | `false` | Enable OpenTelemetry tracing |
| `observability.otel_endpoint` | string | `localhost:4317` | OTLP gRPC endpoint |
| `observability.otel_service_name` | string | `foundry-mcp` | Service name in traces |
| `observability.otel_sample_rate` | float | `1.0` | Trace sampling rate (0.0-1.0) |
| `observability.prometheus_enabled` | bool | `false` | Enable Prometheus metrics |
| `observability.prometheus_port` | int | `0` | HTTP port for /metrics (0=disabled) |
| `observability.prometheus_host` | string | `0.0.0.0` | HTTP server bind address |
| `observability.prometheus_namespace` | string | `foundry_mcp` | Metric name prefix |

### Environment Variables

All TOML settings can be overridden via environment variables:

| Environment Variable | TOML Equivalent |
|---------------------|-----------------|
| `FOUNDRY_MCP_OBSERVABILITY_ENABLED` | `observability.enabled` |
| `FOUNDRY_MCP_OTEL_ENABLED` | `observability.otel_enabled` |
| `FOUNDRY_MCP_OTEL_ENDPOINT` | `observability.otel_endpoint` |
| `FOUNDRY_MCP_OTEL_SERVICE_NAME` | `observability.otel_service_name` |
| `FOUNDRY_MCP_OTEL_SAMPLE_RATE` | `observability.otel_sample_rate` |
| `FOUNDRY_MCP_PROMETHEUS_ENABLED` | `observability.prometheus_enabled` |
| `FOUNDRY_MCP_PROMETHEUS_PORT` | `observability.prometheus_port` |
| `FOUNDRY_MCP_PROMETHEUS_HOST` | `observability.prometheus_host` |
| `FOUNDRY_MCP_PROMETHEUS_NAMESPACE` | `observability.prometheus_namespace` |

## Metrics Catalog

foundry-mcp emits the following Prometheus metrics:

### Tool Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `foundry_mcp_tool_invocations_total` | Counter | tool, status | Total tool invocations |
| `foundry_mcp_tool_duration_seconds` | Histogram | tool | Tool execution duration |
| `foundry_mcp_tool_errors_total` | Counter | tool, error_type | Total tool errors |
| `foundry_mcp_active_operations` | Gauge | operation_type | Currently active operations |

### Resource Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `foundry_mcp_resource_access_total` | Counter | resource_type, action | Total resource accesses |

For the full catalog with examples, see `src/foundry_mcp/core/metrics_registry.py`.

## Grafana Dashboards

Pre-built Grafana dashboards are available in `docs/dashboards/`:

- **Overview** (`foundry-mcp-overview.json`): Server health and request metrics
- **Tools** (`foundry-mcp-tools.json`): Per-tool performance analysis
- **Errors** (`foundry-mcp-errors.json`): Error tracking and analysis

See `docs/dashboards/README.md` for import instructions.

## Troubleshooting

### Observability Not Working

1. **Check dependencies are installed:**
   ```bash
   pip show prometheus-client opentelemetry-sdk
   ```

2. **Verify configuration:**
   ```python
   from foundry_mcp.core.observability import get_observability_status
   print(get_observability_status())
   ```

3. **Check logs for initialization errors:**
   Look for "Failed to initialize OpenTelemetry" or "Failed to initialize Prometheus" messages.

### Metrics Not Appearing

1. **Verify Prometheus port is exposed:**
   ```bash
   curl http://localhost:9090/metrics
   ```

2. **Check `prometheus_port` is non-zero** in configuration.

3. **Verify `observability.enabled = true`** - this is the master switch.

### Traces Not Appearing in Jaeger/Zipkin

1. **Verify OTLP endpoint is reachable:**
   ```bash
   nc -zv localhost 4317
   ```

2. **Check `otel_sample_rate`** - set to `1.0` for debugging.

3. **Ensure traces are flushed** - the server calls `shutdown()` on exit to flush pending spans.

### High Memory Usage

If observability is causing memory issues:

1. Reduce `otel_sample_rate` to sample fewer traces
2. Ensure the OTLP endpoint is healthy (spans queue when endpoint is down)
3. Set `prometheus_port = 0` if not using Prometheus HTTP server

## Programmatic Access

For advanced use cases, access the observability manager directly:

```python
from foundry_mcp.core.observability import get_observability_manager

manager = get_observability_manager()

# Check what's enabled
print(f"Tracing: {manager.is_tracing_enabled()}")
print(f"Metrics: {manager.is_metrics_enabled()}")

# Get a tracer for custom spans
tracer = manager.get_tracer("my-module")
with tracer.start_as_current_span("my-operation"):
    # ... do work
    pass

# Get Prometheus exporter for custom metrics
exporter = manager.get_prometheus_exporter()
exporter.record_tool_invocation("custom-tool", success=True, duration_ms=42.5)
```
