# LLM Integration Guide - foundry-mcp

A comprehensive guide for configuring and using LLM-powered features in foundry-mcp.

Quick setup and environment variable summaries live in [Configuration](../06-configuration.md).
For tool/action listings, see [MCP Tool Reference](../05-mcp-tool-reference.md).

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [CLI Providers](#cli-providers)
- [Provider Management Tools](#provider-management-tools)
- [LLM-Powered Tools](#llm-powered-tools)
- [Graceful Degradation](#graceful-degradation)
- [Multi-Provider Support](#multi-provider-support)
- [Circuit Breaker Resilience](#circuit-breaker-resilience)
- [Feature Flags](#feature-flags)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

foundry-mcp provides LLM-powered features for intelligent spec review and fidelity analysis. These features are designed with resilience in mind, supporting multiple CLI-based LLM providers and graceful degradation when LLM services are unavailable.

### Key Features

- **Multi-provider support**: CLI-based providers (claude, gemini, codex, cursor-agent, opencode)
- **Graceful degradation**: Data-only responses when LLM is unavailable
- **Circuit breaker protection**: Automatic failure handling and recovery
- **External AI tool integration**: cursor-agent, gemini, codex for spec reviews
- **Rate limiting awareness**: Built-in rate limit handling with retry logic

---

## Configuration

### TOML Configuration

Create a `foundry-mcp.toml` file in your project root:

```toml
[consultation]
# Provider priority list - first available wins
# Format: "[cli]transport[:backend/model|:model]"
priority = [
    "[cli]gemini:pro",
    "[cli]claude:opus",
    "[cli]opencode:openai/gpt-5.2",
]

# Per-provider overrides (optional)
[consultation.overrides]
"[cli]opencode:openai/gpt-5.2" = { timeout = 600 }

# Operational settings
default_timeout = 300
max_retries = 2
fallback_enabled = true

[workflow]
mode = "single"               # Execution mode: "single", "autonomous", or "batch"
auto_validate = true          # Automatically run validation after task completion
journal_enabled = true        # Enable journaling of task completions
batch_size = 5                # Number of tasks to execute in batch mode
context_threshold = 85        # Context usage threshold (%) to trigger pause
```

### Environment Variables

Environment variables provide fallback configuration and can override TOML settings:

| Variable | Description | Example |
|----------|-------------|---------|
| `FOUNDRY_MCP_CONSULTATION_PRIORITY` | Comma-separated priority list | `[cli]gemini:pro,[cli]claude:opus` |
| `FOUNDRY_MCP_CONSULTATION_TIMEOUT` | Default timeout in seconds | `300` |
| `FOUNDRY_MCP_CONSULTATION_MAX_RETRIES` | Max retry attempts | `2` |
| `FOUNDRY_MCP_CONSULTATION_RETRY_DELAY` | Delay between retries in seconds | `5.0` |
| `FOUNDRY_MCP_CONSULTATION_FALLBACK_ENABLED` | Enable provider fallback | `true` |
| `FOUNDRY_MCP_CONSULTATION_CACHE_TTL` | Cache TTL in seconds | `3600` |

### Configuration Priority

1. TOML config file (explicit values)
2. `FOUNDRY_MCP_CONSULTATION_*` environment variables
3. Default values

---

## CLI Providers

foundry-mcp uses CLI-based providers that invoke AI tools via subprocess. Available providers:

| Provider | Description | Spec Example |
|----------|-------------|--------------|
| `claude` | Anthropic Claude CLI | `[cli]claude:opus` |
| `gemini` | Google Gemini CLI | `[cli]gemini:pro` |
| `codex` | OpenAI Codex CLI | `[cli]codex` |
| `cursor-agent` | Cursor AI integration | `[cli]cursor-agent:claude-sonnet` |
| `opencode` | OpenCode with backend routing | `[cli]opencode:openai/gpt-5.2` |

---

## Provider Management Tools

foundry-mcp exposes MCP tools for discovering, checking, and invoking LLM providers:

### provider-list

List all registered providers with availability status.

```json
{
  "tool": "provider-list",
  "include_unavailable": false
}
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `include_unavailable` | boolean | No | Include providers that fail availability check (default: `false`) |

**Returns:**
```json
{
  "success": true,
  "data": {
    "providers": [
      {
        "id": "gemini",
        "description": "Google Gemini AI",
        "priority": 100,
        "tags": ["ai", "google"],
        "available": true
      }
    ],
    "available_count": 3,
    "total_count": 5
  }
}
```

### provider-status

Get detailed status for a specific provider.

```json
{
  "tool": "provider-status",
  "provider_id": "gemini"
}
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider_id` | string | Yes | Provider identifier (e.g., `gemini`, `codex`, `cursor-agent`, `claude`, `opencode`) |

**Returns:**
```json
{
  "success": true,
  "data": {
    "provider_id": "gemini",
    "available": true,
    "metadata": {
      "name": "Gemini",
      "version": "2.0",
      "default_model": "gemini-2.0-flash",
      "supported_models": ["..."],
      "documentation_url": "https://ai.google.dev/",
      "tags": ["ai", "google"]
    },
    "capabilities": ["chat", "code_generation", "analysis"],
    "health": {
      "status": "available",
      "reason": null,
      "checked_at": "2025-12-01T12:00:00Z"
    }
  }
}
```

### provider-execute

Execute a prompt through a specified LLM provider.

```json
{
  "tool": "provider-execute",
  "provider_id": "gemini",
  "prompt": "Explain the concept of dependency injection",
  "model": "gemini-2.0-flash",
  "max_tokens": 1000,
  "temperature": 0.7,
  "timeout": 300
}
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider_id` | string | Yes | Provider identifier |
| `prompt` | string | Yes | Prompt text to send to the provider |
| `model` | string | No | Model override (uses provider default if not specified) |
| `max_tokens` | integer | No | Maximum tokens in response |
| `temperature` | float | No | Sampling temperature 0.0-2.0 |
| `timeout` | integer | No | Request timeout in seconds (default: 300) |

**Returns:**
```json
{
  "success": true,
  "data": {
    "provider_id": "gemini",
    "model": "gemini-2.0-flash",
    "content": "Dependency injection is a design pattern...",
    "finish_reason": "stop",
    "token_usage": {
      "prompt_tokens": 10,
      "completion_tokens": 150,
      "total_tokens": 160
    }
  }
}
```

**Error Handling:**
- `UNAVAILABLE`: Provider not configured or available
- `TIMEOUT`: Request exceeded timeout
- `EXECUTION_ERROR`: Provider returned an error

---

## LLM-Powered Tools

foundry-mcp provides six LLM-powered tools:

### spec-review

Run LLM-powered review sessions on specifications.

```json
{
  "tool": "spec-review",
  "spec_id": "feature-auth-001",
  "review_type": "full",
  "tools": "cursor-agent,gemini"
}
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `spec_id` | string | Yes | Specification ID to review |
| `review_type` | string | No | `quick`, `full`, `security`, `feasibility` (default: `quick`) |
| `tools` | string | No | Comma-separated list of review tools |
| `model` | string | No | Override LLM model |
| `dry_run` | boolean | No | Preview without executing |

### review-list-tools

List available review tools and their status.

```json
{
  "tool": "review-list-tools"
}
```

**Returns:** Available AI tools (cursor-agent, gemini, codex) and their availability.

### review-list-plan-tools

Enumerate review toolchains for plan analysis.

```json
{
  "tool": "review-list-plan-tools"
}
```

### spec-review-fidelity

Compare implementation against specification requirements.

```json
{
  "tool": "spec-review-fidelity",
  "spec_id": "feature-auth-001",
  "phase_id": "phase-1",
  "use_ai": true,
  "consensus_threshold": 2
}
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `spec_id` | string | Yes | Specification ID |
| `task_id` | string | No | Review specific task |
| `phase_id` | string | No | Review entire phase |
| `files` | array | No | Review specific files |
| `use_ai` | boolean | No | Enable AI consultation (default: `true`) |
| `ai_tools` | array | No | Specific AI tools to use |
| `consensus_threshold` | integer | No | Models that must agree (default: 2) |
| `incremental` | boolean | No | Only review changed files |

**Rate limit:** 20/hour

---

## Graceful Degradation

When LLM services are unavailable, foundry-mcp automatically falls back to data-only responses.

### How It Works

1. **Detection:** Tools check LLM availability before operations
2. **Fallback:** If unavailable, tools return structured data without AI analysis
3. **Transparency:** Responses include `llm_available: false` indicator
4. **No errors:** Users get useful output even without LLM

### Example Fallback Response

```json
{
  "success": true,
  "data": {
    "spec_id": "feature-auth-001",
    "tasks": ["..."],
    "progress": 75
  },
  "meta": {
    "llm_available": false,
    "fallback_reason": "LLM provider not configured",
    "features_disabled": ["ai_analysis", "suggestions"]
  }
}
```

### Configuring Fallback Behavior

The `llm_data_only_fallback` feature flag controls this behavior:

```python
from foundry_mcp.core.discovery import LLM_FEATURE_FLAGS

# Check if fallback is enabled
fallback_enabled = LLM_FEATURE_FLAGS["llm_data_only_fallback"].default_enabled
```

---

## Multi-Provider Support

foundry-mcp supports using multiple AI tools for enhanced review capabilities.

### External AI Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `cursor-agent` | Cursor AI integration | Code-aware reviews |
| `gemini` | Google Gemini | Broad analysis |
| `codex` | OpenAI Codex | Code generation/review |

### Using Multiple Tools

```bash
# Via CLI
sdd spec-review my-spec-001 --tools cursor-agent,gemini

# Via MCP tool
{
  "tool": "spec-review",
  "spec_id": "my-spec-001",
  "tools": "cursor-agent,gemini,codex"
}
```

### Consensus Mechanism

For fidelity reviews, multiple AI tools can be consulted with a consensus threshold:

```json
{
  "tool": "spec-review-fidelity",
  "spec_id": "feature-auth-001",
  "ai_tools": ["cursor-agent", "gemini"],
  "consensus_threshold": 2
}
```

**Output includes consensus data:**

```json
{
  "consensus": {
    "models_consulted": 3,
    "agreement": "unanimous",
    "confidence": 0.95
  }
}
```

---

## Circuit Breaker Resilience

LLM tools are protected by circuit breakers to prevent cascading failures.

### Configuration

```python
# Default circuit breaker settings for review tools
_review_breaker = CircuitBreaker(
    name="sdd_cli_review",
    failure_threshold=5,      # Opens after 5 consecutive failures
    recovery_timeout=30.0,    # Tries again after 30 seconds
    half_open_max_calls=3,    # Test calls in half-open state
)
```

### States

| State | Behavior |
|-------|----------|
| **Closed** | Normal operation, requests flow through |
| **Open** | Requests fail immediately (circuit tripped) |
| **Half-Open** | Limited test requests to check recovery |

### Timeout Settings

| Operation | Timeout |
|-----------|---------|
| Fast operations | 30 seconds (`FAST_TIMEOUT`) |
| Medium operations | 60 seconds (`MEDIUM_TIMEOUT`) |
| Slow operations (reviews) | 120 seconds (`SLOW_TIMEOUT`) |

### Handling Circuit Breaker Errors

```python
from foundry_mcp.core.resilience import CircuitBreakerError

try:
    result = await spec_review(spec_id)
except CircuitBreakerError as e:
    # Circuit is open, service temporarily unavailable
    logger.warning(f"Service unavailable: {e}")
    # Use fallback behavior
```

---

## Feature Flags

LLM features are controlled by feature flags for gradual rollout and capability negotiation.

### Available Flags

| Flag | Description | State | Default |
|------|-------------|-------|---------|
| `llm_tools` | LLM-powered review | stable | enabled |
| `llm_multi_provider` | Multi-provider AI tool support | stable | enabled |
| `llm_fidelity_review` | AI-powered fidelity review | stable | enabled |
| `llm_data_only_fallback` | Graceful degradation when LLM unavailable | stable | enabled |

### Checking Capabilities

```python
from foundry_mcp.core.discovery import get_llm_capabilities

capabilities = get_llm_capabilities()
# Returns:
# {
#     "llm_tools": {"supported": True, "tools": [...]},
#     "multi_provider": {"supported": True, "providers": [...]},
#     "data_only_fallback": {"supported": True},
#     "feature_flags": {...}
# }
```

### Checking If a Tool Is LLM-Powered

```python
from foundry_mcp.core.discovery import is_llm_tool, get_llm_tool_metadata

if is_llm_tool("spec-review"):
    metadata = get_llm_tool_metadata("spec-review")
    print(f"Category: {metadata.category}")  # "llm"
    print(f"Rate limit: {metadata.rate_limit}")  # "10/hour"
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `LLM provider not configured` | No CLI providers available | Install a supported CLI tool (claude, gemini, codex) |
| `Rate limit exceeded` | Too many requests | Wait for retry-after period |
| `Circuit breaker open` | Too many failures | Wait for recovery timeout |

### Checking LLM Status

```python
from foundry_mcp.tools.unified.review_helpers import _get_llm_status

status = _get_llm_status()
# Returns:
# {"configured": True, "available": True, "providers": [...]}
# or
# {"configured": False, "error": "No AI config available"}
```

---

## Best Practices

### 1. Always Configure Fallback

Ensure your workflows handle LLM unavailability gracefully:

```python
result = await spec_review(spec_id)
if not result.get("meta", {}).get("llm_available", True):
    # LLM was unavailable, handle data-only response
    pass
```

### 2. Use Appropriate Timeouts

LLM operations can be slow. Use appropriate timeouts:

```toml
[consultation]
default_timeout = 300  # Increase for complex operations
```

### 3. Respect Rate Limits

Check rate limits before batch operations:

```python
metadata = get_llm_tool_metadata("spec-review-fidelity")
print(f"Rate limit: {metadata.rate_limit}")
```

### 4. Monitor Circuit Breaker State

Check circuit breaker status for diagnostics:

```python
from foundry_mcp.tools.review import _review_breaker

print(f"Circuit state: {_review_breaker.state}")
print(f"Failure count: {_review_breaker.failure_count}")
```

### 5. Use Consensus for Critical Reviews

For important fidelity reviews, use multiple AI tools with consensus:

```json
{
  "tool": "spec-review-fidelity",
  "ai_tools": ["cursor-agent", "gemini", "codex"],
  "consensus_threshold": 2
}
```

---

## Backlog Cleanup

When transitioning between LLM features or providers:

1. **Clear cached responses:** Remove stale LLM-generated content
2. **Reset circuit breakers:** Restart server after config changes
3. **Verify feature flags:** Check capabilities after updates
4. **Test fallback paths:** Ensure data-only mode works correctly
