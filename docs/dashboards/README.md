# Foundry MCP Grafana Dashboards

Pre-built Grafana dashboards for monitoring foundry-mcp observability metrics.

## Prerequisites

- **Prometheus**: Running and scraping foundry-mcp metrics endpoint
- **Grafana**: Version 9.0+ recommended
- **foundry-mcp**: Configured with Prometheus metrics enabled

### Enabling Metrics in foundry-mcp

Add to your `foundry-mcp.toml`:

```toml
[observability]
enabled = true
prometheus_enabled = true
prometheus_port = 9090  # HTTP port for /metrics endpoint
```

## Available Dashboards

| Dashboard | File | Description |
|-----------|------|-------------|
| Overview | `foundry-mcp-overview.json` | High-level server health, request rates, latencies |
| Tools | `foundry-mcp-tools.json` | Per-tool invocation metrics and performance |
| Errors | `foundry-mcp-errors.json` | Error rates, types, and trends |
| **SLO** | `foundry-mcp-slo.json` | Error budget tracking, burn rates, SLI performance |

## Prometheus Rules

The `prometheus-rules/` directory contains recording and alerting rules for SLO-based monitoring:

| File | Description |
|------|-------------|
| `foundry-mcp-recording-rules.yaml` | Pre-computed SLI metrics (availability, latency, error budget) |
| `foundry-mcp-alerting-rules.yaml` | Multi-burn rate alerts for SLO violations |

See `prometheus-rules/README.md` for installation and configuration details.

## Import Instructions

### Via Grafana UI

1. Open Grafana and navigate to **Dashboards** > **Import**
2. Click **Upload JSON file** and select the dashboard JSON
3. Select your Prometheus datasource
4. Click **Import**

### Via Grafana API

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d @foundry-mcp-overview.json \
  http://localhost:3000/api/dashboards/db
```

## Datasource Configuration

Dashboards expect a Prometheus datasource named `Prometheus`. If your datasource has a different name, update the `datasource` fields in the JSON before importing, or use Grafana's datasource variable feature.

### Prometheus Scrape Config

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'foundry-mcp'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

## Metric Namespace

All foundry-mcp metrics use the `foundry_mcp_` prefix by default. This can be customized via the `prometheus_namespace` configuration option.
