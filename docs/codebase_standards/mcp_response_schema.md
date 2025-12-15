# MCP Response Schema Guide

This document captures the canonical response contract that every Foundry MCP operation must follow. It complements the completed specification [`specs/completed/response-schema-standardization-2025-11-26-001.json`](../../specs/completed/response-schema-standardization-2025-11-26-001.json), the shared helpers in [`src/foundry_mcp/core/responses.py`](../../src/foundry_mcp/core/responses.py), and the best-practice guidance in [docs/mcp_best_practices](../mcp_best_practices/README.md).

## Standard Envelope

All tool responses **must** serialize to the following structure ("response-v2"):

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "meta": {
    "version": "response-v2",
    "request_id": "req_abc123"?,
    "warnings": ["..."]?,
    "pagination": { ... }?,
    "rate_limit": { ... }?,
    "telemetry": { ... }?
  }
}
```

- `success` (bool): Indicates whether the operation completed successfully.
- `data` (object): Operation payload. When no payload exists, send `{}`. For errors, may contain structured error context (see [Error Response Fields](#error-response-fields)).
- `error` (string | null): Populated only when `success` is `false`. Human-readable error description.
- `meta` (object): Always include `{"version": "response-v2"}` and attach optional metadata using the reserved keys listed below.

### Metadata Field Semantics

| Key              | Required | Description |
|------------------|----------|-------------|
| `version`        | YES      | Identifies the response schema version (`response-v2`). |
| `request_id`     | SHOULD   | Correlation identifier propagated through logs/traces. |
| `warnings`       | SHOULD   | Non-fatal issues for successful operations (array of strings). |
| `pagination`     | MAY      | Cursor-based pagination object containing `cursor`, `has_more`, `total_count`, etc. |
| `rate_limit`     | MAY      | Remaining quota, reset time, and retry hints when throttling occurs. |
| `telemetry`      | MAY      | Timing/performance metrics such as `duration_ms` or downstream call counts. |

> Do **not** invent new top-level keys under `data` to convey metadata. Attach operational context through `meta` so every tool shares the same envelope semantics.

### Edge Case Semantics

| Scenario                    | Required Behavior |
|----------------------------|--------------------|
| Empty but successful query | `success: true`, include empty arrays/counts in `data`, `error: null`. |
| Missing resource / invalid input | `success: false`, `data` contains structured error info, descriptive `error` string. |
| Blocked or partial work    | `success: true`, describe state inside `data`, add `meta.warnings` if applicable. |
| Multi-payload operations   | Nest each payload under a named key inside `data` (e.g., `{ "spec": {...}, "tasks": [...] }`). |

## Error Response Fields

When `success` is `false`, the `data` object should contain structured error context to enable machine-readable error handling. The `error_response` helper automatically populates these fields.

### Standard Error Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `error_code` | SHOULD | string | Machine-readable error classification (e.g., `VALIDATION_ERROR`, `NOT_FOUND`, `RATE_LIMIT_EXCEEDED`). Use SCREAMING_SNAKE_CASE. |
| `error_type` | SHOULD | string | Error category for routing/handling (e.g., `validation`, `authorization`, `not_found`, `internal`). |
| `remediation` | SHOULD | string | Actionable guidance for resolving the error. |
| `details` | MAY | object | Nested structure with field-specific or context-specific error info. |

### Error Type Categories

| error_type | HTTP Analog | Description | Retry? |
|------------|-------------|-------------|--------|
| `validation` | 400 | Invalid input data | No, fix input |
| `authentication` | 401 | Invalid or missing credentials | No, re-authenticate |
| `authorization` | 403 | Insufficient permissions | No |
| `not_found` | 404 | Requested resource doesn't exist | No |
| `conflict` | 409 | State conflict (e.g., duplicate) | Maybe, check state |
| `rate_limit` | 429 | Too many requests | Yes, after delay |
| `feature_flag` | 403 | Feature not enabled for client | No, check flag status |
| `internal` | 500 | Server-side error | Yes, with backoff |
| `unavailable` | 503 | Service temporarily unavailable | Yes, with backoff |

### Error Response Example

```json
{
  "success": false,
  "data": {
    "error_code": "VALIDATION_ERROR",
    "error_type": "validation",
    "remediation": "Provide a non-empty spec_id parameter",
    "details": {
      "field": "spec_id",
      "constraint": "required",
      "received": null
    }
  },
  "error": "Validation failed: spec_id is required",
  "meta": {
    "version": "response-v2",
    "request_id": "req_abc123"
  }
}
```

### Common Error Codes

| Error Code | error_type | When to Use |
|------------|------------|-------------|
| `VALIDATION_ERROR` | validation | Generic input validation failure |
| `INVALID_FORMAT` | validation | Malformed input (wrong type, bad JSON) |
| `MISSING_REQUIRED` | validation | Required field not provided |
| `NOT_FOUND` | not_found | Resource doesn't exist |
| `SPEC_NOT_FOUND` | not_found | Specification file not found |
| `TASK_NOT_FOUND` | not_found | Task ID not found in spec |
| `DUPLICATE_ENTRY` | conflict | Resource already exists |
| `CONFLICT` | conflict | State conflict or invalid transition |
| `UNAUTHORIZED` | authentication | Invalid or missing credentials |
| `FORBIDDEN` | authorization | Insufficient permissions |
| `FEATURE_DISABLED` | feature_flag | Feature flag not enabled |
| `RATE_LIMIT_EXCEEDED` | rate_limit | Too many requests |
| `INTERNAL_ERROR` | internal | Unexpected server error |
| `UNAVAILABLE` | unavailable | Service temporarily unavailable |

## Helper Usage

Always leverage the shared helpers and dataclasses to produce responses:

```python
from dataclasses import asdict
from foundry_mcp.core.responses import success_response, error_response

@mcp.tool()
def tool_example(...) -> dict:
    payload = compute_payload(...)
    return asdict(success_response(
        data={"result": payload},
        warnings=payload.warnings,
        pagination=payload.pagination,
        request_id=context.request_id,
    ))
```

### Error Response Helper

For failures, use `error_response()` to include machine-readable context:

```python
return asdict(error_response(
    message="Validation failed: spec_id is required",
    error_code="MISSING_REQUIRED",
    error_type="validation",
    remediation="Provide a non-empty spec_id parameter",
    details={"field": "spec_id", "constraint": "required"},
    request_id=context.request_id,
))
```

**`error_response()` Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `message` | YES | Human-readable error description (populates `error` field) |
| `error_code` | SHOULD | Machine-readable code (e.g., `VALIDATION_ERROR`) |
| `error_type` | SHOULD | Error category (e.g., `validation`, `not_found`) |
| `remediation` | SHOULD | Actionable guidance for resolving the error |
| `data` | MAY | Additional context to merge into `data` object |
| `details` | MAY | Nested error details (field info, constraints) |
| `request_id` | SHOULD | Correlation ID for tracing |
| `rate_limit` | MAY | Rate limit state when applicable |
| `telemetry` | MAY | Timing/performance data captured before failure |
| `meta` | MAY | Additional metadata to merge into `meta` object |

**Example: Not Found Error**

```python
return asdict(error_response(
    message=f"Spec '{spec_id}' not found",
    error_code="SPEC_NOT_FOUND",
    error_type="not_found",
    remediation="Verify the spec ID exists using spec(action=\"list\")", 
    request_id=context.request_id,
))
```

**Example: Feature Flag Disabled**

```python
return asdict(error_response(
    message=f"Feature '{flag_name}' is not enabled",
    error_code="FEATURE_DISABLED",
    error_type="feature_flag",
    data={"feature": flag_name},
    remediation="Contact support to enable this feature or check feature flag configuration",
))
```

**Example: Rate Limit Exceeded**

```python
return asdict(error_response(
    message="Rate limit exceeded: 100 requests per minute",
    error_code="RATE_LIMIT_EXCEEDED",
    error_type="rate_limit",
    data={"retry_after_seconds": 45},
    remediation="Wait 45 seconds before retrying. Consider batching requests.",
    rate_limit={"limit": 100, "remaining": 0, "reset_at": reset_timestamp},
))
```

These helpers guarantee `meta.version` is present and prevent ad-hoc response shapes. Avoid constructing dicts manually.

## Implementation Checklist

1. **Import helpers** (`success_response` / `error_response`) in every tool module.
2. **Return `asdict(...)`** so dataclasses serialize with the standardized keys.
3. **Keep `data` payloads business-focused**; put operational context in `meta`.
4. **Document any additional `meta` semantics** (new pagination fields, telemetry) in tool specs.
5. **Record deviations or streaming quirks** in specs to prevent regressions.

## Testing Requirements

- Unit enforcement lives in [`tests/test_responses.py`](../../tests/test_responses.py); extend it when updating helpers.
- Integration tests such as [`tests/integration/test_mcp_tools.py`](../../tests/integration/test_mcp_tools.py) should assert the envelope for new tools.
- Fixtures and parity harnesses must verify `meta.version == "response-v2"` and any declared metadata keys (`warnings`, `pagination`, etc.).

## Compatibility & Rollout Notes

- The `response_contract_v2` feature flag governs client opt-in. During migrations, continue returning v1 only when explicitly required and document timelines in the spec.
- Feature-flag lifecycles must follow [docs/mcp_best_practices/14-feature-flags.md](../mcp_best_practices/14-feature-flags.md), and metadata such as rate limits should align with [docs/mcp_best_practices/02-envelopes-metadata.md](../mcp_best_practices/02-envelopes-metadata.md).
- Telemetry counters in `foundry_mcp/server.py` rely on consistent envelopes; avoid bypassing the helpers or mutating the serialized dict afterward.

## Related References

- Spec: [`response-schema-standardization-2025-11-26-001`](../../specs/completed/response-schema-standardization-2025-11-26-001.json)
- Helpers: [`src/foundry_mcp/core/responses.py`](../../src/foundry_mcp/core/responses.py)
- Testing/fixtures: [docs/mcp_best_practices/10-testing-fixtures.md](../mcp_best_practices/10-testing-fixtures.md)
- Envelopes & metadata guidance: [docs/mcp_best_practices/02-envelopes-metadata.md](../mcp_best_practices/02-envelopes-metadata.md)
