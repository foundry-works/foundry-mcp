# Troubleshooting

Common issues and solutions for foundry-mcp.

## Table of Contents

- [Setup Issues](#setup-issues)
- [Spec Issues](#spec-issues)
- [Task Issues](#task-issues)
- [MCP/Tool Issues](#mcptool-issues)
- [LLM/AI Issues](#llmai-issues)
- [Test Issues](#test-issues)
- [Performance Issues](#performance-issues)
- [Debugging Tips](#debugging-tips)

---

## Setup Issues

### No specs directory found

**Symptoms:**
- CLI returns `"No specs directory found"`
- Error code: `WORKSPACE_NOT_FOUND`

**Solutions:**

1. Create the specs directory:
   ```bash
   mkdir -p specs/{pending,active,completed,archived}
   ```

2. Set the environment variable:
   ```bash
   export FOUNDRY_MCP_SPECS_DIR=/path/to/specs
   # or
   export SDD_SPECS_DIR=/path/to/specs
   ```

3. Use the CLI option:
   ```bash
   foundry-cli --specs-dir /path/to/specs specs find
   ```

### CLI command not found

**Symptoms:**
- `foundry-cli: command not found`

**Solutions:**

1. Install the package:
   ```bash
   pip install foundry-mcp
   ```

2. Check if it's in PATH:
   ```bash
   which foundry-cli
   python -m foundry_mcp.cli --help
   ```

3. If using a virtual environment, activate it first:
   ```bash
   source venv/bin/activate
   ```

### MCP server won't start

**Symptoms:**
- Server process exits immediately
- Client can't connect

**Solutions:**

1. Check dependencies are installed:
   ```bash
   pip install foundry-mcp[all]
   ```

2. Verify the entry point:
   ```bash
   foundry-mcp --help
   ```

3. Check for port conflicts if using HTTP transport

4. Run with debug logging:
   ```bash
   FOUNDRY_MCP_LOG_LEVEL=DEBUG foundry-mcp
   ```

---

## Spec Issues

### Spec not found

**Symptoms:**
- Error: `"Specification not found: {spec_id}"`
- Error code: `SPEC_NOT_FOUND`

**Solutions:**

1. List available specs:
   ```bash
   foundry-cli specs find
   foundry-cli specs find --status all
   ```

2. Check the spec ID format - should match `{name}-{date}-{sequence}`:
   ```
   my-feature-2025-01-15-001
   ```

3. Verify the spec file exists:
   ```bash
   ls specs/*/my-feature*.json
   ```

### Spec validation errors

**Symptoms:**
- Validation fails with diagnostics
- Cannot save spec changes

**Solutions:**

1. Run validation with details:
   ```bash
   foundry-cli validate check my-spec
   ```

2. Auto-fix common issues:
   ```bash
   foundry-cli validate fix my-spec --dry-run  # preview
   foundry-cli validate fix my-spec            # apply
   ```

3. Check for common issues:
   - Missing required fields (id, title, status)
   - Invalid status values
   - Orphaned nodes (no parent)
   - Invalid parent references

### Circular dependency detected

**Symptoms:**
- Error: `"Circular dependency detected"`
- Error code: `CIRCULAR_DEPENDENCY`

**Solutions:**

1. Analyze the dependency graph:
   ```bash
   foundry-cli validate analyze-deps my-spec
   ```

2. Look for cycles in `blocked_by` / `blocks` relationships

3. Remove one dependency to break the cycle:
   ```bash
   # Via MCP tool
   {"action": "remove-dependency", "spec_id": "my-spec", "task_id": "task-a", "target_id": "task-b"}
   ```

### Spec won't complete

**Symptoms:**
- Error: `"Cannot complete spec with pending tasks"`
- Error code: `INCOMPLETE_TASKS`

**Solutions:**

1. Check progress:
   ```bash
   foundry-cli validate stats my-spec
   ```

2. List pending tasks:
   ```bash
   foundry-cli specs query-tasks my-spec --status pending
   ```

3. Force completion if needed:
   ```bash
   foundry-cli lifecycle complete my-spec --force
   ```

---

## Task Issues

### Task not found

**Symptoms:**
- Error: `"Task not found: {task_id}"`
- Error code: `TASK_NOT_FOUND`

**Solutions:**

1. List tasks in the spec:
   ```bash
   foundry-cli specs query-tasks my-spec
   ```

2. Check the task ID format (e.g., `task-1-2`, `phase-1`)

3. View the hierarchy:
   ```bash
   # Via MCP tool
   {"action": "hierarchy", "spec_id": "my-spec"}
   ```

### Cannot start task (blocked)

**Symptoms:**
- Error: `"Task is blocked"`
- Error code: `BLOCKED_TASK` or `INCOMPLETE_DEPS`

**Solutions:**

1. Check what's blocking the task:
   ```bash
   foundry-cli tasks check-complete my-spec task-1-2
   ```

2. Complete blocking tasks first:
   ```bash
   foundry-cli tasks complete my-spec blocking-task --note "Done"
   ```

3. View all blocked tasks:
   ```bash
   foundry-cli specs list-blockers my-spec
   ```

### Invalid status transition

**Symptoms:**
- Error: `"Invalid status transition"`
- Error code: `INVALID_TRANSITION`

**Solutions:**

Valid transitions:
- `pending` → `in_progress`
- `pending` → `blocked`
- `in_progress` → `completed`
- `in_progress` → `blocked`
- `blocked` → `pending`
- `blocked` → `in_progress`

You cannot go directly from `pending` to `completed` or from `completed` to any other status.

---

## MCP/Tool Issues

### Tool or action not found

**Symptoms:**
- Error: `"Unknown action"`
- Error code: `INVALID_ACTION`

**Solutions:**

1. Check available actions for the tool:
   ```bash
   # Via MCP tool
   {"action": "schema", "tool_name": "spec"}
   ```

2. Refer to [MCP Tool Reference](05-mcp-tool-reference.md)

3. Check `mcp/capabilities_manifest.json` for the source of truth

### MCP server starts but tools fail

**Symptoms:**
- Server running but tool calls error
- Unexpected responses

**Solutions:**

1. Run health check:
   ```json
   {"action": "liveness"}
   ```

2. Check capabilities:
   ```json
   {"action": "capabilities"}
   ```

3. Verify client is connecting correctly

4. Check server logs with debug level

### Response format unexpected

**Symptoms:**
- Missing fields in response
- Response doesn't match expected structure

**Solutions:**

1. All responses use the standard envelope:
   ```json
   {"success": true, "data": {...}, "error": null, "meta": {...}}
   ```

2. Check `meta.version` should be `"response-v2"`

3. See [Response Envelope Guide](concepts/response-envelope.md)

---

## LLM/AI Issues

### LLM features not working

**Symptoms:**
- AI reviews fail
- Provider errors

**Solutions:**

1. Check LLM status:
   ```json
   {"action": "llm-status"}
   ```

2. Verify configuration in `foundry-mcp.toml`:
   ```toml
   [consultation]
   default_provider = "gemini"

   [consultation.providers.gemini]
   api_key_env = "GEMINI_API_KEY"
   ```

3. Check environment variables:
   ```bash
   echo $GEMINI_API_KEY
   ```

4. See [LLM Configuration Guide](guides/llm-configuration.md)

### AI provider timeout

**Symptoms:**
- Error: `"Provider timeout"`
- Error code: `PROVIDER_TIMEOUT`

**Solutions:**

1. Increase timeout in config:
   ```toml
   [consultation]
   default_timeout = 360
   ```

2. Use CLI option:
   ```bash
   foundry-cli review spec my-spec --ai-timeout 600
   ```

3. Use a smaller scope:
   ```bash
   foundry-cli review fidelity my-spec --task task-1-2
   ```

### No AI providers available

**Symptoms:**
- Error: `"No providers available"`
- Error code: `AI_NO_PROVIDER`

**Solutions:**

1. Configure at least one provider in `foundry-mcp.toml`

2. Check provider list:
   ```json
   {"action": "list"}
   ```

3. Verify API keys are set:
   ```bash
   env | grep -i api_key
   ```

---

## Test Issues

### pytest not found

**Symptoms:**
- Error: `"pytest not found"`
- Error code: `PYTEST_NOT_FOUND`

**Solutions:**

```bash
pip install pytest
pip install pytest-cov  # for coverage
```

### Test discovery fails

**Symptoms:**
- Error: `"Test discovery failed"`
- Error code: `TEST_DISCOVERY_FAILED`

**Solutions:**

1. Check for collection errors:
   ```bash
   pytest --collect-only
   ```

2. Look for import errors in test files

3. Verify test file naming (`test_*.py` or `*_test.py`)

### Test timeout

**Symptoms:**
- Error: `"Test run timed out"`
- Error code: `TIMEOUT`

**Solutions:**

1. Increase timeout:
   ```bash
   foundry-cli test run --timeout 600
   ```

2. Use a smaller target:
   ```bash
   foundry-cli test run tests/unit/
   ```

3. Use fail-fast:
   ```bash
   foundry-cli test run --fail-fast
   ```

---

## Performance Issues

### Validation or review timeouts

**Symptoms:**
- Operations timing out
- Slow response times

**Solutions:**

1. Increase timeout settings:
   ```toml
   [consultation]
   default_timeout = 600
   ```

2. Use quick review instead of full:
   ```bash
   foundry-cli review spec my-spec --type quick
   ```

3. Review smaller scope (single task/phase)

4. Disable optional LLM steps when not needed

### Large spec files slow

**Symptoms:**
- Slow load times
- High memory usage

**Solutions:**

1. Check spec size:
   ```bash
   foundry-cli validate stats my-spec
   ```

2. Consider splitting into multiple specs

3. Archive completed phases to separate specs

---

## Debugging Tips

### Enable debug logging

```bash
FOUNDRY_MCP_LOG_LEVEL=DEBUG foundry-cli specs find
```

### Check response details

```bash
foundry-cli specs find | jq '.'
```

### Validate JSON manually

```bash
cat specs/active/my-spec.json | jq '.'
```

### Run health check

```bash
# CLI
foundry-cli --help

# MCP
{"action": "check", "include_details": true}
```

### Check configuration

```bash
cat foundry-mcp.toml
env | grep -i foundry
env | grep -i sdd
```

---

## Error Code Quick Reference

| Code | Type | Common Cause |
|------|------|--------------|
| `VALIDATION_ERROR` | validation | Invalid input parameters |
| `SPEC_NOT_FOUND` | not_found | Wrong spec ID or missing file |
| `TASK_NOT_FOUND` | not_found | Wrong task ID |
| `BLOCKED_TASK` | conflict | Task has incomplete dependencies |
| `CIRCULAR_DEPENDENCY` | conflict | Dependency cycle exists |
| `PROVIDER_TIMEOUT` | unavailable | AI provider too slow |
| `TIMEOUT` | internal | Operation exceeded time limit |

See [Error Codes Reference](reference/error-codes.md) for the complete list.

---

## Still Stuck?

1. Run with debug logging:
   ```bash
   FOUNDRY_MCP_LOG_LEVEL=DEBUG foundry-cli <command>
   ```

2. Check the error response for `remediation` field

3. Verify your configuration:
   ```bash
   cat foundry-mcp.toml
   ```

4. Review the documentation:
   - [CLI Command Reference](04-cli-command-reference.md)
   - [MCP Tool Reference](05-mcp-tool-reference.md)
   - [Configuration Guide](06-configuration.md)

---

## Deep Research Resilience Issues

### Research task timed out

**Symptoms:**
- Status response shows `is_timed_out: true`
- Error message mentions timeout

**Solutions:**

1. Increase the workflow timeout:
   ```toml
   [research]
   deep_research_timeout = 900.0  # 15 minutes
   ```

2. Pass explicit timeout in API call:
   ```json
   {"action": "deep-research", "query": "...", "task_timeout": 1200}
   ```

3. Reduce scope to complete faster:
   - Decrease `max_iterations` (default: 3)
   - Decrease `max_sub_queries` (default: 5)
   - Decrease `max_sources_per_query` (default: 5)

### Research task appears stale

**Symptoms:**
- Status response shows `is_stale: true`
- `last_heartbeat_at` is more than 5 minutes ago

**Causes:**
- Provider is slow to respond
- Network issues between server and LLM provider
- Provider rate limiting

**Solutions:**

1. Check provider status and availability
2. Review `last_heartbeat_at` to see when last activity occurred
3. Consider cancelling and retrying with different provider:
   ```json
   {"action": "deep-research", "research_id": "...", "deep_research_action": "cancel"}
   ```

### Cancellation not working

**Symptoms:**
- Cancel action returns success but task continues
- Task shows as cancelled but still consuming resources

**Solutions:**

1. Cancellation uses two-phase approach:
   - First, sets cooperative cancellation flag
   - Then, forces cancellation after 5 seconds

2. If task is stuck in provider call, it will complete current operation before checking cancellation flag

3. Check task status to confirm cancellation:
   ```json
   {"action": "deep-research-status", "research_id": "..."}
   ```

### Partial results after crash

**Symptoms:**
- Research was interrupted mid-workflow
- Status shows partial progress

**Solutions:**

1. Check status to see what was completed:
   ```json
   {"action": "deep-research-status", "research_id": "..."}
   ```

2. Resume from last checkpoint:
   ```json
   {"action": "deep-research", "research_id": "...", "deep_research_action": "continue"}
   ```

3. If resume fails, start new research - state is persisted after each phase
