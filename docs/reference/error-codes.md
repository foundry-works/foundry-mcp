# Error Codes Reference

This document lists all error codes returned by foundry-mcp tools and CLI commands.

## Error Response Structure

When an operation fails, the response includes:

```json
{
  "success": false,
  "data": {
    "error_code": "VALIDATION_ERROR",
    "error_type": "validation",
    "remediation": "Provide a non-empty spec_id parameter",
    "details": { ... }
  },
  "error": "Human-readable error message",
  "meta": { "version": "response-v2" }
}
```

| Field | Description |
|-------|-------------|
| `error_code` | Machine-readable code (SCREAMING_SNAKE_CASE) |
| `error_type` | Error category for routing/handling |
| `remediation` | Actionable guidance for resolution |
| `details` | Context-specific error information |

---

## Error Types

| Type | HTTP Analog | Description | Retry? |
|------|-------------|-------------|--------|
| `validation` | 400 | Invalid input data | No, fix input |
| `authentication` | 401 | Invalid or missing credentials | No, re-authenticate |
| `authorization` | 403 | Insufficient permissions | No |
| `not_found` | 404 | Requested resource doesn't exist | No |
| `conflict` | 409 | State conflict (e.g., duplicate) | Maybe, check state |
| `rate_limit` | 429 | Too many requests | Yes, after delay |
| `feature_flag` | 403 | Feature not enabled | No, check flag status |
| `internal` | 500 | Server-side error | Yes, with backoff |
| `unavailable` | 503 | Service temporarily unavailable | Yes, with backoff |

---

## Error Codes by Category

### Validation Errors

| Code | Type | Description | Remediation |
|------|------|-------------|-------------|
| `VALIDATION_ERROR` | validation | Generic input validation failure | Check input parameters against schema |
| `INVALID_FORMAT` | validation | Malformed input (wrong type, bad JSON) | Verify data types and JSON syntax |
| `MISSING_REQUIRED` | validation | Required field not provided | Add the required parameter |
| `INVALID_ACTION` | validation | Unknown action parameter | Check available actions for the tool |
| `INVALID_STATUS` | validation | Invalid status value | Use valid status: `pending`, `in_progress`, `completed`, `blocked` |
| `INVALID_OPTIONS` | validation | Mutually exclusive options used together | Use only one of the conflicting options |
| `INVALID_TRANSITION` | validation | Invalid status transition | Check valid transitions (e.g., `pending` → `in_progress`) |

### Not Found Errors

| Code | Type | Description | Remediation |
|------|------|-------------|-------------|
| `NOT_FOUND` | not_found | Generic resource not found | Verify the resource identifier |
| `SPEC_NOT_FOUND` | not_found | Specification file not found | Use `specs list` to see available specs |
| `TASK_NOT_FOUND` | not_found | Task ID not found in spec | Use `tasks list` to see available tasks |
| `PHASE_NOT_FOUND` | not_found | Phase ID not found in spec | Check phase IDs in the spec |
| `TEMPLATE_NOT_FOUND` | not_found | Template doesn't exist | Use `template list` to see available templates |
| `WORKSPACE_NOT_FOUND` | not_found | Workspace/specs directory not found | Set `--specs-dir` or `FOUNDRY_SPECS_DIR` |
| `JOURNAL_NOT_FOUND` | not_found | Journal entry not found | Check journal entry ID |
| `BACKUP_NOT_FOUND` | not_found | Backup file not found | Use `history` action to see available backups |

### State/Conflict Errors

| Code | Type | Description | Remediation |
|------|------|-------------|-------------|
| `CONFLICT` | conflict | Generic state conflict | Check current state before retrying |
| `DUPLICATE_ENTRY` | conflict | Resource already exists | Use a different identifier or update existing |
| `ALREADY_EXISTS` | conflict | Spec/task already exists | Choose a different name or ID |
| `CIRCULAR_DEPENDENCY` | conflict | Dependency would create cycle | Review dependency graph |
| `BLOCKED_TASK` | conflict | Task is blocked and cannot proceed | Resolve blocker first |
| `INCOMPLETE_DEPS` | conflict | Dependencies not completed | Complete dependent tasks first |
| `SPEC_LOCKED` | conflict | Spec is locked for editing | Wait for lock release or use force |

### Lifecycle Errors

| Code | Type | Description | Remediation |
|------|------|-------------|-------------|
| `INVALID_LIFECYCLE_TRANSITION` | validation | Invalid lifecycle state change | Check valid transitions for current state |
| `INCOMPLETE_TASKS` | conflict | Cannot complete spec with pending tasks | Complete all tasks or use `--force` |
| `ALREADY_ACTIVE` | conflict | Spec already in active state | No action needed |
| `ALREADY_COMPLETED` | conflict | Spec already completed | Use archive or revert if needed |

### AI/Provider Errors

| Code | Type | Description | Remediation |
|------|------|-------------|-------------|
| `AI_NOT_AVAILABLE` | unavailable | AI consultation layer not available | Check installation |
| `AI_NO_PROVIDER` | unavailable | No AI providers configured | Configure an AI provider |
| `AI_CONSULTATION_ERROR` | internal | AI consultation failed | Check provider config and retry |
| `PROVIDER_ERROR` | internal | LLM provider returned error | Check API keys and quotas |
| `PROVIDER_TIMEOUT` | unavailable | Provider request timed out | Increase timeout or retry |

### Test Errors

| Code | Type | Description | Remediation |
|------|------|-------------|-------------|
| `TEST_FAILED` | internal | Test execution failed | Inspect output and fix failing tests |
| `TEST_DISCOVERY_FAILED` | internal | Test discovery failed | Check for collection errors |
| `PYTEST_NOT_FOUND` | internal | pytest not installed | Run `pip install pytest` |
| `TIMEOUT` | internal | Operation timed out | Increase timeout or reduce scope |
| `CONSULT_FAILED` | internal | Test consultation failed | Check AI tool configuration |

### System Errors

| Code | Type | Description | Remediation |
|------|------|-------------|-------------|
| `INTERNAL_ERROR` | internal | Unexpected server error | Report issue with details |
| `UNAVAILABLE` | unavailable | Service temporarily unavailable | Retry with backoff |
| `SPEC_LOAD_ERROR` | internal | Failed to load spec file | Check JSON validity |
| `SPEC_SAVE_ERROR` | internal | Failed to save spec file | Check file permissions |
| `FILE_WRITE_ERROR` | internal | Failed to write file | Check disk space and permissions |

### Permission/Auth Errors

| Code | Type | Description | Remediation |
|------|------|-------------|-------------|
| `UNAUTHORIZED` | authentication | Invalid or missing credentials | Provide valid credentials |
| `FORBIDDEN` | authorization | Insufficient permissions | Check user permissions |
| `FEATURE_DISABLED` | feature_flag | Feature not enabled | Contact support or enable feature |
| `RATE_LIMIT_EXCEEDED` | rate_limit | Too many requests | Wait and retry |

---

## Handling Errors

### Retryable Errors

These errors may succeed on retry:

- `INTERNAL_ERROR` - Retry with exponential backoff
- `UNAVAILABLE` - Retry after delay
- `RATE_LIMIT_EXCEEDED` - Retry after `retry_after_seconds`
- `PROVIDER_TIMEOUT` - Retry with increased timeout
- `CONFLICT` - Retry after checking state

### Non-Retryable Errors

These errors require fixing the input:

- `VALIDATION_ERROR` - Fix input parameters
- `NOT_FOUND` - Use correct resource ID
- `INVALID_TRANSITION` - Check current state
- `CIRCULAR_DEPENDENCY` - Restructure dependencies

### Example: Error Handling in Code

```python
result = tool_call(...)

if not result["success"]:
    error_type = result["data"].get("error_type")
    error_code = result["data"].get("error_code")

    if error_type == "rate_limit":
        retry_after = result["data"].get("retry_after_seconds", 60)
        time.sleep(retry_after)
        # retry...
    elif error_type == "validation":
        remediation = result["data"].get("remediation")
        print(f"Fix input: {remediation}")
    elif error_type in ("internal", "unavailable"):
        # exponential backoff retry
        pass
```

---

## Related

- [Response Envelope Guide](../concepts/response-envelope.md) - Full response format
- [Troubleshooting](../07-troubleshooting.md) - Common issues and solutions
- [MCP Tool Reference](../05-mcp-tool-reference.md) - Tool documentation
