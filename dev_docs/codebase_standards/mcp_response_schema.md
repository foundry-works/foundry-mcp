# MCP Response Schema Guide

This document captures the canonical response contract that every Foundry MCP operation must follow. It complements the completed specification [`specs/completed/response-schema-standardization-2025-11-26-001.json`](../../specs/completed/response-schema-standardization-2025-11-26-001.json), the shared helpers in [`src/foundry_mcp/core/responses.py`](../../src/foundry_mcp/core/responses.py), and the best-practice guidance in [dev_docs/mcp_best_practices](../mcp_best_practices/README.md).

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
    "warning_details": [{ "code": "...", "severity": "...", "message": "..." }]?,
    "pagination": { ... }?,
    "rate_limit": { ... }?,
    "telemetry": { ... }?,
    "content_fidelity": "full" | "partial" | "summary" | "reference_only"?,
    "content_fidelity_schema_version": "1.0"?,
    "dropped_content_ids": ["..."]?,
    "content_archive_hashes": { ... }?
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
| `content_fidelity` | MAY    | Content fidelity level indicating completeness of the response (see [Content Fidelity Metadata](#content-fidelity-metadata)). |
| `content_fidelity_schema_version` | MAY | Schema version for content fidelity metadata (e.g., `"1.0"`). |
| `dropped_content_ids` | MAY  | Array of content identifiers that were dropped due to size constraints. |
| `content_archive_hashes` | MAY | Object mapping archive identifiers to content hashes for retrieval. |
| `warning_details` | MAY      | Structured warning objects with severity and context (see [Warning Details](#warning-details)). |

> Do **not** invent new top-level keys under `data` to convey metadata. Attach operational context through `meta` so every tool shares the same envelope semantics.

### Edge Case Semantics

| Scenario                    | Required Behavior |
|----------------------------|--------------------|
| Empty but successful query | `success: true`, include empty arrays/counts in `data`, `error: null`. |
| Missing resource / invalid input | `success: false`, `data` contains structured error info, descriptive `error` string. |
| Blocked or partial work    | `success: true`, describe state inside `data`, add `meta.warnings` if applicable. |
| Multi-payload operations   | Nest each payload under a named key inside `data` (e.g., `{ "spec": {...}, "tasks": [...] }`). |

## Content Fidelity Metadata

When responses may be truncated, summarized, or have content dropped due to token limits or size constraints, include content fidelity metadata in `meta` to inform consumers about response completeness.

### Content Fidelity Schema

```json
{
  "meta": {
    "version": "response-v2",
    "content_fidelity_schema_version": "1.0",
    "content_fidelity": "full" | "partial" | "summary" | "reference_only",
    "dropped_content_ids": ["finding-003", "source-015"],
    "content_archive_hashes": {
      "archive-001": "sha256:abc123..."
    }
  }
}
```

### Content Fidelity Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content_fidelity_schema_version` | string | SHOULD (when fidelity < full) | Schema version for content fidelity metadata. Current version: `"1.0"`. |
| `content_fidelity` | string | SHOULD (when fidelity < full) | Level of content completeness in the response. |
| `dropped_content_ids` | array\<string\> | MAY | Identifiers of content items that were omitted. Enables targeted retrieval. |
| `content_archive_hashes` | object | MAY | Map of archive IDs to content hashes for retrieving dropped content. |

### Content Fidelity Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `full` | Complete response with all content included | Default when no truncation occurs |
| `partial` | Some content omitted but structure preserved | Large responses exceeding soft limits |
| `summary` | Condensed representation of full content | Token-constrained contexts |
| `reference_only` | Only identifiers/references, no content bodies | Extreme token constraints |

### Content Fidelity Example

Response with partial fidelity due to dropped findings:

```json
{
  "success": true,
  "data": {
    "task_id": "task-001",
    "findings": [
      {"id": "finding-001", "title": "Primary result", "content": "..."},
      {"id": "finding-002", "title": "Secondary result", "content": "..."}
    ],
    "total_findings": 5
  },
  "error": null,
  "meta": {
    "version": "response-v2",
    "content_fidelity_schema_version": "1.0",
    "content_fidelity": "partial",
    "dropped_content_ids": ["finding-003", "finding-004", "finding-005"],
    "content_archive_hashes": {
      "findings-archive": "sha256:e3b0c44298fc1c149afbf4c8996fb924..."
    },
    "warnings": ["3 findings omitted due to token limits"]
  }
}
```

### Content Retrieval Pattern

When `dropped_content_ids` is present, consumers can retrieve omitted content:

1. Check `dropped_content_ids` for missing item identifiers
2. Use `content_archive_hashes` to verify archive availability
3. Call appropriate retrieval endpoint with the archive hash or content IDs

## Warning Details

For structured warnings with severity and context beyond simple string messages, use `warning_details` alongside or instead of the `warnings` array.

### Warning Details Schema

```json
{
  "meta": {
    "version": "response-v2",
    "warnings": ["3 findings omitted due to token limits"],
    "warning_details": [
      {
        "code": "CONTENT_TRUNCATED",
        "severity": "info",
        "message": "3 findings omitted due to token limits",
        "context": {
          "dropped_count": 3,
          "total_count": 5,
          "reason": "token_limit_exceeded"
        }
      }
    ]
  }
}
```

### Warning Detail Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | string | SHOULD | Machine-readable warning classification (e.g., `CONTENT_TRUNCATED`, `STALE_CACHE`). |
| `severity` | string | SHOULD | Warning severity level: `info`, `warning`, `error`. |
| `message` | string | YES | Human-readable warning description. |
| `context` | object | MAY | Additional context specific to the warning type. |

### Warning Severity Levels

| Severity | Description | Consumer Action |
|----------|-------------|-----------------|
| `info` | Informational, no action needed | Log/display as appropriate |
| `warning` | Potential issue, consider action | Evaluate context and decide |
| `error` | Significant issue, action recommended | Address before proceeding |

### Standard Warning Codes

| Code | Severity | Description |
|------|----------|-------------|
| `CONTENT_TRUNCATED` | info | Response content was truncated due to size limits |
| `STALE_CACHE` | warning | Cached data may be outdated |
| `PARTIAL_FAILURE` | warning | Some sub-operations failed but overall succeeded |
| `DEPRECATED_FIELD` | info | Response includes deprecated fields |
| `RATE_LIMIT_APPROACHING` | warning | Approaching rate limit threshold |
| `FALLBACK_USED` | info | Primary source unavailable, fallback used |

### Warning Details Example

```json
{
  "success": true,
  "data": {
    "results": [...]
  },
  "error": null,
  "meta": {
    "version": "response-v2",
    "warnings": [
      "3 sources failed to respond",
      "Cache data is 2 hours old"
    ],
    "warning_details": [
      {
        "code": "PARTIAL_FAILURE",
        "severity": "warning",
        "message": "3 sources failed to respond",
        "context": {
          "failed_sources": ["source-a", "source-b", "source-c"],
          "successful_sources": 7,
          "total_sources": 10
        }
      },
      {
        "code": "STALE_CACHE",
        "severity": "warning",
        "message": "Cache data is 2 hours old",
        "context": {
          "cache_age_seconds": 7200,
          "max_freshness_seconds": 3600
        }
      }
    ]
  }
}
```

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
- Feature-flag lifecycles must follow [dev_docs/mcp_best_practices/14-feature-flags.md](../mcp_best_practices/14-feature-flags.md), and metadata such as rate limits should align with [dev_docs/mcp_best_practices/02-envelopes-metadata.md](../mcp_best_practices/02-envelopes-metadata.md).
- Telemetry counters in `foundry_mcp/server.py` rely on consistent envelopes; avoid bypassing the helpers or mutating the serialized dict afterward.

## Related References

- Spec: [`response-schema-standardization-2025-11-26-001`](../../specs/completed/response-schema-standardization-2025-11-26-001.json)
- Helpers: [`src/foundry_mcp/core/responses.py`](../../src/foundry_mcp/core/responses.py)
- Testing/fixtures: [dev_docs/mcp_best_practices/10-testing-fixtures.md](../mcp_best_practices/10-testing-fixtures.md)
- Envelopes & metadata guidance: [dev_docs/mcp_best_practices/02-envelopes-metadata.md](../mcp_best_practices/02-envelopes-metadata.md)
