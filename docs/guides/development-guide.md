# Development Guide

A comprehensive guide for developers contributing to foundry-mcp.

## Table of Contents

- [Project Overview](#project-overview)
- [Development Setup](#development-setup)
- [Architecture](#architecture)
- [Code Organization](#code-organization)
- [Adding New Tools](#adding-new-tools)
- [Response Contract](#response-contract)
- [Testing](#testing)
- [CLI Development](#cli-development)
- [Common Workflows](#common-workflows)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

foundry-mcp is an MCP (Model Context Protocol) server that enables AI assistants to manage spec-driven development workflows. It provides 40+ tools for:

- **Spec Management** — Create, validate, and lifecycle specs
- **Task Operations** — Track progress, dependencies, and blockers
- **Code Intelligence** — Query codebase documentation and call graphs
- **Testing** — Run and discover tests with pytest integration
- **LLM Features** — AI-powered reviews, documentation, and PR generation

### Key Design Principles

1. **Transport-Agnostic Core** — Business logic lives in `foundry_mcp.core.*`, shared by both MCP server and CLI
2. **JSON-First Output** — All responses follow a standardized envelope for LLM consumption
3. **Graceful Degradation** — LLM features work without API keys, falling back to structural analysis
4. **Spec-Driven Workflow** — Changes are planned in specifications before implementation

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- pip or uv

### Installation

```bash
# Clone the repository
git clone https://github.com/tylerburleigh/foundry-mcp.git
cd foundry-mcp

# Install in development mode with test dependencies
pip install -e ".[test]"

# Verify installation
python -c "import foundry_mcp; print('foundry-mcp installed')"
```

### Environment Configuration

Create a `foundry-mcp.toml` in your project root (optional):

```toml
[workspace]
specs_dir = "./specs"

[logging]
level = "DEBUG"
structured = true

[workflow]
mode = "single"
auto_validate = true
journal_enabled = true
```

Or use environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `FOUNDRY_MCP_SPECS_DIR` | Path to specs directory | Auto-detected |
| `FOUNDRY_MCP_LOG_LEVEL` | Logging level | INFO |
| `FOUNDRY_MCP_WORKFLOW_MODE` | Execution mode | single |
| `FOUNDRY_MCP_CONFIG_FILE` | Path to TOML config | Auto-detected |

### Running the Server

```bash
# Start MCP server (production mode)
foundry-mcp

# Start with debug logging
FOUNDRY_MCP_LOG_LEVEL=DEBUG foundry-mcp

# Run from source directly
python -m foundry_mcp.server
```

### MCP Development Mode

```bash
# Interactive testing with MCP CLI
mcp dev foundry-mcp

# Or with uvx
uvx --from mcp dev foundry-mcp
```

---

## Architecture

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (server.py)                   │
│              FastMCP + Tool Registration                    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Tools Layer (tools/*.py)                  │
│         MCP Tool Implementations + Response Formatting       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Core Layer (core/*.py)                    │
│         Business Logic (transport-agnostic)                  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    CLI Layer (cli/*.py)                      │
│         Click Commands + JSON Output                         │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **Server** | `server.py` | MCP server creation, tool registration, entry point |
| **Tools** | `tools/*.py` | MCP tool implementations, parameter validation, response formatting |
| **Core** | `core/*.py` | Pure business logic, no transport dependencies |
| **CLI** | `cli/*.py` | Command-line interface using Click |

### Key Principle

**All business logic must live in `core/`**. Tools and CLI commands are thin wrappers that call core functions and format responses.

---

## Code Organization

```
src/foundry_mcp/
├── __init__.py
├── server.py              # MCP server entry point
├── config.py              # Configuration management
│
├── core/                  # Business logic (transport-agnostic)
│   ├── spec.py            # Spec file operations
│   ├── task.py            # Task operations
│   ├── journal.py         # Journal management
│   ├── lifecycle.py       # Spec lifecycle transitions
│   ├── validation.py      # Spec validation
│   ├── responses.py       # Response helpers (success_response, error_response)
│   ├── feature_flags.py   # Feature flag management
│   ├── pagination.py      # Cursor-based pagination
│   ├── observability.py   # Logging and metrics
│   └── ...
│
├── tools/                 # MCP tool implementations
│   ├── tasks.py           # Task-related tools
│   ├── validation.py      # Validation tools
│   ├── lifecycle.py       # Lifecycle tools
│   ├── journal.py         # Journal tools
│   ├── testing.py         # Test runner tools
│   ├── review.py          # LLM review tools
│   └── ...
│
├── cli/                   # Command-line interface
│   ├── main.py            # CLI entry point
│   ├── output.py          # JSON output formatting
│   ├── commands/          # Click command modules
│   │   ├── tasks.py
│   │   ├── lifecycle.py
│   │   └── ...
│   └── ...
│
├── resources/             # MCP resources (foundry:// URIs)
│   └── specs.py
│
└── prompts/               # MCP prompts (workflow templates)
    └── workflows.py
```

### Tool Module Structure

Each tool module follows this pattern:

```python
# tools/example.py

from dataclasses import asdict
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool

def register_example_tools(mcp, config):
    """Register example-related tools with the MCP server."""

    @mcp.tool()
    @canonical_tool("example-do-thing")
    def example_do_thing(param: str) -> dict:
        """
        Brief description of what this tool does.

        Args:
            param: Description of the parameter

        Returns:
            JSON object with result data
        """
        # Call core logic
        from foundry_mcp.core.example import do_thing
        result = do_thing(param)

        # Return standardized response
        return asdict(success_response(data={"result": result}))
```

---

## Adding New Tools

### Step 1: Implement Core Logic

Create or extend a module in `core/`:

```python
# core/example.py

def do_thing(param: str) -> dict:
    """
    Pure business logic with no transport dependencies.

    Args:
        param: Input parameter

    Returns:
        Result dictionary
    """
    # Implementation here
    return {"processed": param}
```

### Step 2: Create Tool Wrapper

Create or extend a module in `tools/`:

```python
# tools/example.py

from dataclasses import asdict
from typing import Optional

from foundry_mcp.core.responses import (
    success_response,
    error_response,
    validation_error,
    not_found_error,
)
from foundry_mcp.core.naming import canonical_tool

def register_example_tools(mcp, config):
    """Register example tools with the MCP server."""

    @mcp.tool()
    @canonical_tool("example-do-thing")
    def example_do_thing(
        param: str,
        optional_param: Optional[str] = None,
    ) -> dict:
        """
        Brief description (shown in tool discovery).

        More detailed description of what this tool does,
        when to use it, and what it returns.

        Args:
            param: Required parameter description
            optional_param: Optional parameter description

        Returns:
            JSON object with:
            - processed: The processed result
            - count: Number of items processed
        """
        # Validate inputs
        if not param:
            return asdict(validation_error(
                "param is required",
                field="param",
                remediation="Provide a non-empty param value",
            ))

        try:
            # Call core logic
            from foundry_mcp.core.example import do_thing
            result = do_thing(param)

            # Return success
            return asdict(success_response(
                data={
                    "processed": result["processed"],
                    "count": result.get("count", 0),
                },
            ))

        except FileNotFoundError:
            return asdict(not_found_error("Example", param))
        except Exception as e:
            return asdict(error_response(
                f"Failed to process: {e}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
            ))
```

### Step 3: Register Tools

Add registration call in `server.py`:

```python
from foundry_mcp.tools.example import register_example_tools

def create_server(config: Optional[ServerConfig] = None) -> FastMCP:
    # ... existing code ...

    # Register tools
    register_example_tools(mcp, config)  # Add this line
```

### Step 4: Add Tests

Create tests in `tests/unit/`:

```python
# tests/unit/test_example.py

import pytest
from foundry_mcp.core.example import do_thing

def test_do_thing_basic():
    result = do_thing("input")
    assert result["processed"] == "input"

def test_do_thing_empty_input():
    with pytest.raises(ValueError):
        do_thing("")
```

### Naming Conventions

Follow the [naming conventions](../codebase_standards/naming-conventions.md):

| Surface | Name | Examples |
|--------|------|----------|
| Unified tool router | `spec` | `spec(action="list")`, `spec(action="validate")` |
| Unified tool router | `task` | `task(action="next")`, `task(action="prepare")` |
| Unified tool router | `lifecycle` | `lifecycle(action="activate")`, `lifecycle(action="archive")` |
| Unified tool router | `journal` | `journal(action="add")`, `journal(action="list")` |
| Unified tool router | `test` | `test(action="run", preset="full")`, `test(action="discover")` |
| Unified tool router | `review` | `review(action="spec")`, `review(action="fidelity")` |
| Unified tool router | `code` | `code(action="find-function")`, `code(action="trace")` |
| Unified tool router | `server` | `server(action="tools")`, `server(action="schema", tool_name="spec")` |

---

## Response Contract

All tool responses **must** use the standardized envelope:

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "meta": {
    "version": "response-v2",
    "request_id": "req_abc123",
    "warnings": ["..."],
    "pagination": { ... }
  }
}
```

### Using Response Helpers

Always use helpers from `core/responses.py`:

```python
from dataclasses import asdict
from foundry_mcp.core.responses import (
    success_response,
    error_response,
    validation_error,
    not_found_error,
    rate_limit_error,
)

# Success response
return asdict(success_response(
    data={"result": "value"},
    warnings=["Non-fatal warning"],
    pagination={"cursor": "abc", "has_more": True},
))

# Validation error
return asdict(validation_error(
    "Invalid email format",
    field="email",
    remediation="Use format: user@domain.com",
))

# Not found error
return asdict(not_found_error("Spec", "my-spec-id"))

# Generic error
return asdict(error_response(
    "Something went wrong",
    error_code="INTERNAL_ERROR",
    error_type="internal",
))
```

### Error Types

| error_type | HTTP Analog | When to Use | Retry? |
|------------|-------------|-------------|--------|
| `validation` | 400 | Invalid input | No |
| `not_found` | 404 | Resource missing | No |
| `conflict` | 409 | State conflict | Maybe |
| `rate_limit` | 429 | Too many requests | Yes |
| `internal` | 500 | Server error | Yes |
| `unavailable` | 503 | Service down | Yes |

See [Response Schema Guide](../codebase_standards/mcp_response_schema.md) for complete documentation.

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_spec.py

# Run tests matching pattern
pytest -k "test_validation"

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Stop on first failure
pytest -x

# Show print statements
pytest -s
```

### Test Presets

| Preset | Use Case | Command Equivalent |
|--------|----------|-------------------|
| `quick` | Fast development feedback | `pytest -m "not slow" -x --timeout=60` |
| `full` | Complete test run | `pytest -v --timeout=300` |
| `unit` | Unit tests only | `pytest -m unit -v --timeout=120` |
| `integration` | Integration tests only | `pytest -m integration -v --timeout=300` |
| `smoke` | Quick verification | `pytest -m smoke -x --timeout=30` |

### Writing Tests

```python
# tests/unit/test_example.py

import pytest
import tempfile
from pathlib import Path

@pytest.fixture
def temp_specs_dir():
    """Create temporary specs directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        specs_dir = Path(tmpdir) / "specs"
        for status in ["pending", "active", "completed", "archived"]:
            (specs_dir / status).mkdir(parents=True)
        yield specs_dir

def test_example_success(temp_specs_dir):
    """Test successful operation."""
    # Arrange
    # ...

    # Act
    result = some_function()

    # Assert
    assert result["success"] is True
    assert "data" in result

@pytest.mark.slow
def test_expensive_operation():
    """Mark slow tests so they can be skipped during quick runs."""
    # ...

@pytest.mark.integration
def test_requires_external_service():
    """Mark integration tests that need external dependencies."""
    # ...
```

### Test Directory Structure

```
tests/
├── unit/                    # Unit tests for core modules
│   ├── test_spec.py
│   ├── test_task.py
│   ├── test_validation.py
│   └── ...
├── integration/             # Integration tests
│   ├── test_mcp_smoke.py
│   ├── test_mcp_tools.py
│   └── ...
└── fixtures/                # Test fixtures
    └── golden/              # Golden fixture files
```

See [Testing Guide](testing.md) for comprehensive testing documentation.

---

## CLI Development

### CLI Architecture

The CLI uses Click and outputs JSON only:

```python
# cli/commands/example.py

import click
from foundry_mcp.cli.output import output_json, output_error
from foundry_mcp.core.example import do_thing

@click.command()
@click.argument("param")
@click.option("--dry-run", is_flag=True, help="Preview without executing")
def example_command(param: str, dry_run: bool):
    """Brief description of the command."""
    try:
        result = do_thing(param, dry_run=dry_run)
        output_json({
            "success": True,
            "data": result,
            "error": None,
        })
    except Exception as e:
        output_error(str(e))
```

### CLI Output Rules

1. **JSON-only output** — No ANSI colors, progress bars, or human formatting
2. **Same response contract** — Use the same envelope as MCP tools
3. **Stderr for errors** — Non-JSON diagnostics go to stderr
4. **Exit codes** — 0 for success, non-zero for failures

### Session Context & Transcript Access

`session context` parses Claude transcript JSONL files to estimate token budgets.
Because those transcripts live outside the repo, access is opt-in:

- Pass `--transcript-dir <path>` to point at an explicit directory of `.jsonl` files
- Set `FOUNDRY_MCP_ALLOW_TRANSCRIPTS=1` or pass `--allow-home-transcripts` to scan
  `~/.claude/projects` derived paths
- Without either control, the command returns `TRANSCRIPTS_DISABLED` to honor
  the trust-boundary guidance in `docs/mcp_best_practices/08-security-trust-boundaries.md`

All transcript scans continue to produce JSON envelopes and respect the
`MEDIUM_TIMEOUT` budget so that exponential backoff never exceeds the declared limit.

### Registering Commands

```python
# cli/main.py

from foundry_mcp.cli.commands.example import example_command

@click.group()
def cli():
    """Foundry CLI - Spec-driven development tools."""
    pass

cli.add_command(example_command)
```

---

## Common Workflows

### Adding a New Feature

1. **Create a spec** — Document the feature in `specs/pending/`
2. **Activate the spec** — Move to `specs/active/` when starting work
3. **Implement core logic** — Add to `core/*.py`
4. **Create tool wrapper** — Add to `tools/*.py`
5. **Register tools** — Update `server.py`
6. **Add tests** — Both unit and integration
7. **Update docs** — Response contract, tool discovery, guides
8. **Complete the spec** — Move to `specs/completed/`

### Debugging Issues

```bash
# Enable debug logging
FOUNDRY_MCP_LOG_LEVEL=DEBUG foundry-mcp

# Check configuration
python -c "from foundry_mcp.config import get_config; print(get_config())"

# Verify spec directory structure
ls -la specs/{pending,active,completed,archived}

# Run specific test with verbose output
pytest tests/unit/test_spec.py -v -s

# Check for import errors
python -c "from foundry_mcp.server import create_server; print('OK')"
```

### Code Review Checklist

- [ ] Uses standardized response envelope via helpers
- [ ] Validates inputs early with descriptive errors
- [ ] Handles errors gracefully with proper error types
- [ ] Includes appropriate logging
- [ ] Has unit test coverage
- [ ] Follows naming conventions
- [ ] Updates relevant documentation

---

## Best Practices

### General

1. **Keep tools thin** — Business logic belongs in `core/`, not `tools/`
2. **Validate early** — Check inputs before calling core functions
3. **Use response helpers** — Never construct response dicts manually
4. **Document thoroughly** — Tool docstrings appear in discovery
5. **Test in isolation** — Each test should be independent

### Response Handling

1. **Always return `success`** — Even for empty results
2. **Distinguish empty vs. not found** — Empty results are `success: true`
3. **Include remediation** — Error messages should be actionable
4. **Use warnings for non-fatal issues** — Don't fail on recoverable problems

### Performance

1. **Paginate large results** — Use cursor-based pagination
2. **Cache expensive operations** — Use `core/cache.py` utilities
3. **Mark slow tests** — Use `@pytest.mark.slow` decorator
4. **Avoid blocking I/O** — Use async patterns for external calls

### Security

1. **Validate all inputs** — Never trust user data
2. **Sanitize paths** — Prevent directory traversal
3. **Mask sensitive data** — API keys, tokens in logs
4. **Respect trust boundaries** — See [Security Guide](../mcp_best_practices/08-security-trust-boundaries.md)

---

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: foundry_mcp` | Package not installed | `pip install -e .` |
| `No specs directory found` | Missing specs path | Set `FOUNDRY_MCP_SPECS_DIR` |
| `Spec not found: {id}` | Wrong spec ID or folder | Check spec exists in status folders |
| `Invalid spec structure` | Malformed JSON | Run `spec(action="validate")` |
| `pytest not found` | Missing test dependency | `pip install -e ".[test]"` |

### Getting Help

1. Check the [Testing Guide](testing.md) for test-related issues
2. Review [MCP Best Practices](../mcp_best_practices/README.md) for patterns
3. Search existing issues on GitHub
4. Create a new issue with reproduction steps

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [Testing Guide](testing.md) | Complete testing documentation |
| [Response Schema](../codebase_standards/mcp_response_schema.md) | Response contract details |
| [Naming Conventions](../codebase_standards/naming-conventions.md) | Tool naming standards |
| [MCP Best Practices](../mcp_best_practices/README.md) | Industry patterns |
| [Architecture](../architecture/adr-001-cli-architecture.md) | CLI/MCP architecture decision record |
| [LLM Configuration](llm-configuration.md) | LLM provider setup |

---

*Last updated: 2025-11-28*
