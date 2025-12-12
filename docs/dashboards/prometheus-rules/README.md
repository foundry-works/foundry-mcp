# Prometheus Rules for foundry-mcp

This directory contains Prometheus recording and alerting rules for SLO-based monitoring of foundry-mcp.

## Files

| File | Description |
|------|-------------|
| `foundry-mcp-recording-rules.yaml` | Pre-computed SLI metrics for efficient querying |
| `foundry-mcp-alerting-rules.yaml` | Multi-burn rate alerts for SLO violations |

## SLO Definitions

| SLO | Target | Error Budget | Window |
|-----|--------|--------------|--------|
| Availability | 99.5% | 0.5% | 30 days |
| Latency (P99) | < 5s | 1% violations | 30 days |
| Error Rate | < 0.5% | 0.5% | 30 days |

## Installation

### 1. Add to Prometheus Configuration

```yaml
# prometheus.yml
rule_files:
  - /path/to/foundry-mcp-recording-rules.yaml
  - /path/to/foundry-mcp-alerting-rules.yaml
```

### 2. Reload Prometheus

```bash
# Signal reload
kill -HUP $(pidof prometheus)

# Or via API (if enabled)
curl -X POST http://localhost:9090/-/reload
```

### 3. Verify Rules Loaded

```bash
# Check rule groups
curl http://localhost:9090/api/v1/rules | jq '.data.groups[].name'
```

## Alert Severity Levels

### Critical (Severity: critical)
- **Burn Rate**: 14.4x (consumes budget in ~2 days)
- **Windows**: 5m AND 1h
- **Action**: Page immediately, any time
- **Examples**: `FoundryMCP_AvailabilityCritical`, `FoundryMCP_Unhealthy`

### Warning (Severity: warning)
- **Burn Rate**: 6x (consumes budget in ~5 days)
- **Windows**: 5m AND 6h
- **Action**: Page during business hours
- **Examples**: `FoundryMCP_AvailabilityWarning`, `FoundryMCP_LatencyWarning`

### Ticket (Severity: ticket)
- **Burn Rate**: 3x (consumes budget in ~10 days)
- **Windows**: 6h AND 24h
- **Action**: Create ticket, no page
- **Examples**: `FoundryMCP_AvailabilityTicket`

## Recording Rules

### SLI Metrics

```promql
# Availability at different windows
foundry_mcp:sli:availability:rate5m
foundry_mcp:sli:availability:rate1h
foundry_mcp:sli:availability:rate6h
foundry_mcp:sli:availability:rate24h
foundry_mcp:sli:availability:rate30d

# Latency metrics
foundry_mcp:sli:latency_p99:rate5m
foundry_mcp:sli:latency_p99:rate1h
foundry_mcp:sli:latency_good:rate5m   # fraction under 5s

# Error rate
foundry_mcp:sli:error_rate:rate5m
foundry_mcp:sli:error_rate:rate1h
```

### Error Budget Metrics

```promql
# Error budget consumption (0-1, >1 means budget exhausted)
foundry_mcp:error_budget:availability_consumed
foundry_mcp:error_budget:availability_remaining
foundry_mcp:error_budget:latency_consumed
foundry_mcp:error_budget:latency_remaining
```

### Burn Rate Metrics

```promql
# Burn rate at different windows
foundry_mcp:burn_rate:availability:5m
foundry_mcp:burn_rate:availability:1h
foundry_mcp:burn_rate:availability:6h
foundry_mcp:burn_rate:availability:24h
```

### Per-Tool Metrics

```promql
# Request rate per tool
foundry_mcp:tool:request_rate:5m

# Error rate per tool
foundry_mcp:tool:error_rate:5m

# Latency per tool
foundry_mcp:tool:latency_p99:5m
foundry_mcp:tool:latency_p50:5m
```

### Manifest Budget Metrics

```promql
# Unified tool manifest size (token estimate)
foundry_mcp:manifest_tokens:unified
foundry_mcp:manifest_tool_count:unified
foundry_mcp:manifest_tokens:budget_ratio

# Feature-flag state exported as gauges
foundry_mcp:feature_flag:unified_manifest
```

## Alertmanager Configuration

Example routing configuration for alertmanager:

```yaml
# alertmanager.yml
route:
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true

    - match:
        severity: warning
      receiver: 'slack-warning'
      group_wait: 5m

    - match:
        severity: ticket
      receiver: 'jira-ticket'
      group_wait: 1h

receivers:
  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '<pagerduty-key>'

  - name: 'slack-warning'
    slack_configs:
      - api_url: '<slack-webhook>'
        channel: '#platform-alerts'

  - name: 'jira-ticket'
    webhook_configs:
      - url: 'http://jira-webhook/create-ticket'
```

## Dashboard Integration

These rules are designed to work with the SLO Grafana dashboard:
- `foundry-mcp-slo.json` - SLO tracking dashboard

Import the dashboard and configure the Prometheus data source to visualize:
- Error budget consumption over time
- Burn rate trends
- SLI performance by window
- Per-tool breakdown

## Customization

### Adjusting SLO Targets

To change the 99.5% availability target:

1. Update recording rules: Change `0.005` (error budget) in burn rate calculations
2. Update alerting rules: Adjust thresholds accordingly
3. Update dashboard panels if using custom SLO targets

### Adding Custom Alerts

Follow the pattern in `foundry-mcp-alerting-rules.yaml`:

```yaml
- alert: MyCustomAlert
  expr: |
    # Your PromQL expression
  for: 5m
  labels:
    severity: warning
    team: my-team
  annotations:
    summary: "Brief description"
    description: "Detailed description with {{ $value }}"
    runbook_url: "https://..."
```

## Troubleshooting

### No Data for SLI Metrics

1. Verify Prometheus metrics are being scraped:
   ```promql
   foundry_mcp_tool_invocations_total
   ```

2. Check recording rules are loaded:
   ```bash
   curl http://localhost:9090/api/v1/rules?type=record
   ```

### Alerts Not Firing

1. Check alert rules are loaded:
   ```bash
   curl http://localhost:9090/api/v1/rules?type=alert
   ```

2. Test the alert expression manually in Prometheus UI

3. Verify `for` duration has elapsed

### High Cardinality

If metrics have too many label combinations:
- Consider aggregating by fewer labels
- Use recording rules to pre-aggregate
- Review label cardinality limits
