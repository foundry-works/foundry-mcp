# 7. Graceful Degradation & Error Semantics

> Distinguish between hard failures and recoverable issues with actionable error messages.

## Overview

Error handling in MCP tools should be predictable and informative. Clients need to understand whether to retry, what went wrong, and how to fix it.

## Requirements

### MUST

- **Distinguish hard failures from warnings** - use `success=false` only for failures
- **Provide actionable error messages** - include what went wrong and how to fix it
- **Use consistent error structure** - same format across all tools
- **Never expose internal details** - no stack traces, SQL, or system paths

### SHOULD

- **Classify errors by type** - validation, authorization, resource, system
- **Include error codes** - machine-readable classification
- **Suggest remediation** - what the client can do
- **Log full details server-side** - keep abbreviated version for response

### MAY

- **Include retry hints** - when and how to retry
- **Provide documentation links** - for complex error resolution
- **Support error localization** - for user-facing messages

## Error Classification

### Error Categories

| Category | HTTP Analog | Description | Retry? |
|----------|-------------|-------------|--------|
| `validation` | 400 | Invalid input | No, fix input |
| `authentication` | 401 | Invalid credentials | No, re-authenticate |
| `authorization` | 403 | Insufficient permissions | No |
| `not_found` | 404 | Resource doesn't exist | No |
| `conflict` | 409 | State conflict | Maybe, check state |
| `rate_limit` | 429 | Too many requests | Yes, after delay |
| `internal` | 500 | Server error | Yes, with backoff |
| `unavailable` | 503 | Service unavailable | Yes, with backoff |

### Error Response Structure

```json
{
    "success": false,
    "data": {
        "error_code": "VALIDATION_ERROR",
        "error_type": "validation",
        "details": {
            "field": "email",
            "constraint": "format",
            "message": "Invalid email format"
        },
        "remediation": "Provide a valid email address (e.g., user@example.com)"
    },
    "error": "Validation failed: invalid email format",
    "meta": {
        "version": "response-v2",
        "request_id": "req_abc123"
    }
}
```

## Error vs Warning Decision Tree

```
Operation completed successfully?
├── YES: success=true
│   └── Any non-fatal issues?
│       ├── YES: Add to meta.warnings
│       └── NO: No warnings needed
└── NO: success=false
    └── Set descriptive error message
```

### Warnings (success=true)

Use warnings for issues that didn't prevent the operation:

```json
{
    "success": true,
    "data": {
        "processed": 97,
        "results": [...]
    },
    "error": null,
    "meta": {
        "version": "response-v2",
        "warnings": [
            "3 items skipped: insufficient permissions",
            "Results may be stale: cache age 5 minutes"
        ]
    }
}
```

## Warning Taxonomy

Standard warning codes enable machine-readable classification of non-fatal issues. Use these codes in `meta.warning_details[].code` for structured warnings.

### Warning Code Registry

| Code | Severity | Description | When to Use |
|------|----------|-------------|-------------|
| `CONTENT_TRUNCATED` | info | Response content was truncated to fit token/size limits | Large responses exceeding soft limits |
| `CONTENT_DROPPED` | info | Specific content items were omitted entirely | Individual items dropped for size |
| `PRIORITY_SUMMARIZED` | info | Low-priority content replaced with summaries | Tiered content compression applied |
| `SUMMARY_PROVIDER_FAILED` | warning | AI summarization failed, using fallback | Summarization service unavailable |
| `TOKEN_BUDGET_FLOORED` | warning | Token budget hit minimum threshold | Budget too low for meaningful content |
| `LIMITS_DEFAULTED` | info | Using default limits (config not found) | Missing or invalid limit configuration |
| `ARCHIVE_WRITE_FAILED` | warning | Failed to archive dropped content for retrieval | Content may not be recoverable |
| `PROTECTED_OVERFLOW` | warning | Protected content exceeded allocated budget | Essential content squeezed other content |
| `STATE_MIGRATION_RECOVERED` | info | State recovered from older schema version | Automatic migration applied |
| `TOKEN_COUNT_ESTIMATE_USED` | info | Actual token count unavailable, using estimate | Tokenizer not available |

### Warning Severity Definitions

| Severity | Description | Consumer Action |
|----------|-------------|-----------------|
| `info` | Informational notice, operation succeeded fully | Log for debugging; no user action needed |
| `warning` | Potential issue that may affect results | Consider displaying to user; evaluate impact |
| `error` | Significant issue, results may be degraded | Alert user; consider retry or alternative |

### Per-Phase Warning Matrix

Different processing phases emit specific warnings. Use this matrix to understand warning origins:

| Phase | Possible Warnings | Trigger Condition |
|-------|-------------------|-------------------|
| **Token Estimation** | `TOKEN_COUNT_ESTIMATE_USED` | Tokenizer unavailable; heuristic used |
| **Budget Allocation** | `TOKEN_BUDGET_FLOORED`, `LIMITS_DEFAULTED` | Budget below minimum or config missing |
| **Content Prioritization** | `PROTECTED_OVERFLOW` | Protected content exceeds allocation |
| **Summarization** | `SUMMARY_PROVIDER_FAILED`, `PRIORITY_SUMMARIZED` | AI provider error or tiered compression |
| **Content Assembly** | `CONTENT_TRUNCATED`, `CONTENT_DROPPED` | Final content exceeds output limits |
| **Archival** | `ARCHIVE_WRITE_FAILED` | Storage write failure |
| **State Management** | `STATE_MIGRATION_RECOVERED` | Loaded state from older schema version |

### Warning vs Error Decision Table

Use this table to determine whether an issue should be a warning (success=true) or an error (success=false):

| Scenario | Warning or Error? | Rationale |
|----------|-------------------|-----------|
| Content truncated but core results present | **Warning** | Primary operation succeeded |
| All content dropped, nothing to return | **Error** | Cannot provide meaningful response |
| Summarization failed, raw content used | **Warning** | Fallback provides usable result |
| Token budget too low to process anything | **Error** | Cannot complete operation |
| Archive write failed, content still returned | **Warning** | Primary response succeeded |
| Archive write failed, content not returned | **Error** | Content unrecoverable |
| Config missing, using safe defaults | **Warning** | Operation proceeds with defaults |
| Config invalid, cannot determine behavior | **Error** | Cannot safely proceed |
| State migration succeeded automatically | **Warning** | Operation completed with recovery |
| State migration failed, data corrupted | **Error** | Cannot recover valid state |
| Some items processed, others failed | **Warning** | Partial success (see Partial Success) |
| All items failed to process | **Error** | Complete failure |

### Warning Response Example

```json
{
    "success": true,
    "data": {
        "task_id": "task-001",
        "findings": [...],
        "total_findings": 15,
        "returned_findings": 10
    },
    "error": null,
    "meta": {
        "version": "response-v2",
        "content_fidelity": "partial",
        "warnings": [
            "5 findings omitted due to token limits",
            "Using estimated token counts"
        ],
        "warning_details": [
            {
                "code": "CONTENT_DROPPED",
                "severity": "info",
                "message": "5 findings omitted due to token limits",
                "context": {
                    "dropped_count": 5,
                    "dropped_ids": ["finding-11", "finding-12", "finding-13", "finding-14", "finding-15"],
                    "reason": "token_budget_exceeded"
                }
            },
            {
                "code": "TOKEN_COUNT_ESTIMATE_USED",
                "severity": "info",
                "message": "Using estimated token counts",
                "context": {
                    "estimation_method": "character_ratio",
                    "accuracy_estimate": "±10%"
                }
            }
        ]
    }
}
```

### Errors (success=false)

Use errors when the primary operation failed:

```json
{
    "success": false,
    "data": {},
    "error": "Database connection failed after 3 retries",
    "meta": {
        "version": "response-v2"
    }
}
```

## Actionable Error Messages

### Good Error Messages

```python
# Validation error
{
    "error": "Invalid email format: 'not-an-email'",
    "data": {
        "error_code": "INVALID_EMAIL",
        "remediation": "Provide email in format: user@domain.com"
    }
}

# Resource not found
{
    "error": "User 'usr_999' not found",
    "data": {
        "error_code": "USER_NOT_FOUND",
        "remediation": "Verify the user ID exists. List users with list_users()."
    }
}

# Rate limit
{
    "error": "Rate limit exceeded: 100 requests per minute",
    "data": {
        "error_code": "RATE_LIMIT_EXCEEDED",
        "retry_after_seconds": 45,
        "remediation": "Wait 45 seconds before retrying. Consider batching requests."
    }
}

# Permission error
{
    "error": "Insufficient permissions to delete project 'proj_123'",
    "data": {
        "error_code": "PERMISSION_DENIED",
        "required_permission": "project:delete",
        "remediation": "Request 'project:delete' permission from project owner."
    }
}
```

### Bad Error Messages

```python
# Too vague
{"error": "Operation failed"}

# Exposes internals
{"error": "NullPointerException at UserService.java:142"}

# No guidance
{"error": "Invalid request"}

# Technical jargon
{"error": "FK constraint violation on users_projects_fk"}
```

## Error Code Registry

Define standard error codes for your MCP server:

```python
from enum import Enum

class ErrorCode(str, Enum):
    # Validation errors (1xxx)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_FORMAT = "INVALID_FORMAT"
    MISSING_REQUIRED = "MISSING_REQUIRED"
    VALUE_OUT_OF_RANGE = "VALUE_OUT_OF_RANGE"

    # Authentication errors (2xxx)
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"

    # Authorization errors (3xxx)
    PERMISSION_DENIED = "PERMISSION_DENIED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"

    # Resource errors (4xxx)
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"

    # Rate limiting (5xxx)
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # System errors (9xxx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    DEPENDENCY_FAILED = "DEPENDENCY_FAILED"
```

## Error Helper Functions

```python
def validation_error(
    message: str,
    field: str = None,
    details: dict = None,
    remediation: str = None
) -> ErrorResponse:
    """Create a validation error response."""
    data = {
        "error_code": "VALIDATION_ERROR",
        "error_type": "validation"
    }
    if field:
        data["field"] = field
    if details:
        data["details"] = details
    if remediation:
        data["remediation"] = remediation

    return error_response(error=message, data=data)

def not_found_error(
    resource_type: str,
    resource_id: str,
    remediation: str = None
) -> ErrorResponse:
    """Create a not found error response."""
    return error_response(
        error=f"{resource_type} '{resource_id}' not found",
        data={
            "error_code": "RESOURCE_NOT_FOUND",
            "error_type": "not_found",
            "resource_type": resource_type,
            "resource_id": resource_id,
            "remediation": remediation or f"Verify the {resource_type.lower()} ID exists."
        }
    )

def rate_limit_error(
    limit: int,
    period: str,
    retry_after_seconds: int
) -> ErrorResponse:
    """Create a rate limit error response."""
    return error_response(
        error=f"Rate limit exceeded: {limit} requests per {period}",
        data={
            "error_code": "RATE_LIMIT_EXCEEDED",
            "error_type": "rate_limit",
            "limit": limit,
            "period": period,
            "retry_after_seconds": retry_after_seconds,
            "remediation": f"Wait {retry_after_seconds} seconds before retrying."
        }
    )
```

## Partial Success Handling

When some items succeed and others fail:

```python
@mcp.tool()
def bulk_delete(item_ids: List[str]) -> dict:
    """Delete multiple items."""
    succeeded = []
    failed = []

    for item_id in item_ids:
        try:
            db.delete(item_id)
            succeeded.append(item_id)
        except NotFoundError:
            failed.append({"id": item_id, "reason": "not found"})
        except PermissionError:
            failed.append({"id": item_id, "reason": "permission denied"})

    # Determine overall success
    if not failed:
        return asdict(success_response(
            data={"deleted": succeeded}
        ))
    elif not succeeded:
        return asdict(error_response(
            error=f"Failed to delete all {len(failed)} items",
            data={"failed": failed}
        ))
    else:
        # Partial success - still success=true with warnings
        return asdict(success_response(
            data={
                "deleted": succeeded,
                "failed": failed
            },
            warnings=[f"{len(failed)} of {len(item_ids)} items could not be deleted"]
        ))
```

## Anti-Patterns

### Don't: Mix Success and Error States

```python
# Bad: success=true but error set
{"success": true, "error": "Something went wrong"}

# Bad: success=false but no error message
{"success": false, "error": null}

# Good: Consistent state
{"success": true, "error": null, "data": {...}}
{"success": false, "error": "Descriptive message", "data": {}}
```

### Don't: Expose Stack Traces

```python
# Bad: Stack trace in response
{
    "error": "Traceback (most recent call last):\n  File \"app.py\", line 42..."
}

# Good: User-friendly message (log full trace server-side)
{
    "error": "An internal error occurred. Request ID: req_abc123"
}
```

### Don't: Use Generic Messages

```python
# Bad: Generic
{"error": "Error"}
{"error": "Something went wrong"}
{"error": "Invalid input"}

# Good: Specific
{"error": "Email 'invalid' is not a valid email address"}
{"error": "User 'usr_999' not found in project 'proj_123'"}
{"error": "Field 'count' must be between 1 and 1000, got 5000"}
```

## Related Documents

- [Envelopes & Metadata](./02-envelopes-metadata.md) - Error response structure
- [Validation & Input Hygiene](./04-validation-input-hygiene.md) - Validation errors
- [Resilience Patterns](./12-timeout-resilience.md) - Retry strategies

---

**Navigation:** [← Pagination & Streaming](./06-pagination-streaming.md) | [Index](./README.md) | [Next: Security & Trust Boundaries →](./08-security-trust-boundaries.md)
