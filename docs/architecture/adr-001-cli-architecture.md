# ADR-001: CLI Architecture

**Status:** Implemented
**Date:** 2025-11-28
**Supersedes:** N/A

## Context

The foundry-mcp project provides MCP (Model Context Protocol) tools for spec-driven development. This ADR defines the architecture for the native CLI implementation within foundry-mcp.

### Architecture

- **MCP Server:** `foundry_mcp.server` - FastMCP-based server with 80+ tools
- **Core Logic:** `foundry_mcp.core.*` - Shared business logic (spec, task, journal, etc.)
- **MCP Tools:** `foundry_mcp.tools.*` - MCP tool implementations calling core logic
- **Native CLI:** `foundry_mcp.cli` - Click-based CLI calling core logic directly

### Design Goals

1. Provide native CLI for standalone usage
2. Share core logic between MCP and CLI surfaces
3. Maintain separation of concerns (transport vs business logic)
4. Enable independent testing of CLI without MCP server

## Key Decision: JSON-Only Output

**Decision:** The SDD CLI outputs JSON exclusively. No `--json` flag, no `--verbose`, no `--quiet`, no human-readable mode.

**Rationale:**
- **Primary consumers are AI coding assistants** (Claude, Cursor, Codex, etc.)
- **AI agents parse structured data best** - no regex/pattern matching needed
- **Consistent output format** = reliable integration
- **Eliminates complexity** of dual output modes and verbosity levels
- **Humans can pipe through `jq`** if they need to inspect output

**Implications:**
- `cli/output.py` always emits JSON - no conditional formatting
- No `--json` flag needed (everything is already JSON)
- No verbosity flags (`-v`, `--verbose`, `--quiet`)
- Simpler code, fewer edge cases, better reliability

**Example output:**
```json
{
  "success": true,
  "data": {"task_id": "task-1-1", "status": "completed"},
  "error": null
}
```

## Decision

### Package Boundaries

```
foundry_mcp/
├── core/                    # Shared business logic (UNCHANGED)
│   ├── spec.py              # Spec file operations
│   ├── task.py              # Task operations
│   ├── journal.py           # Journal operations
│   ├── lifecycle.py         # Lifecycle state machine
│   ├── validation.py        # Input validation
│   ├── responses.py         # Response envelope helpers
│   └── ...
│
├── tools/                   # MCP tool implementations (UNCHANGED)
│   ├── tasks.py             # @mcp.tool decorators
│   ├── lifecycle.py         # Lifecycle MCP tools
│   └── ...
│
├── cli/                     # NEW: CLI runtime
│   ├── __init__.py          # CLI package init
│   ├── main.py              # Entry point (click app)
│   ├── commands/            # Command groups
│   │   ├── __init__.py
│   │   ├── spec.py          # sdd spec-* commands
│   │   ├── task.py          # sdd task-* commands
│   │   ├── journal.py       # sdd journal-* commands
│   │   ├── lifecycle.py     # sdd activate/complete/archive
│   │   ├── validation.py    # sdd validate/fix commands
│   │   ├── docs.py          # sdd doc-* commands
│   │   ├── test.py          # sdd test-* commands
│   │   └── review.py        # sdd review-* commands
│   ├── output.py            # JSON/text output formatting
│   ├── config.py            # CLI-specific config (work_mode, etc.)
│   └── context.py           # Session/context tracking (CLI-only)
│
├── server.py                # MCP server (UNCHANGED)
└── config.py                # Shared configuration
```

### Namespace Structure

| Namespace | Purpose | Dependencies |
|-----------|---------|--------------|
| `foundry_mcp.core` | Pure business logic | Standard library, pydantic |
| `foundry_mcp.tools` | MCP tool decorators | `foundry_mcp.core`, `mcp` |
| `foundry_mcp.cli` | CLI runtime | `foundry_mcp.core`, `click` |
| `foundry_mcp.cli.commands` | Click command groups | `foundry_mcp.cli`, `foundry_mcp.core` |

### Key Principles

1. **Core is transport-agnostic**: `foundry_mcp.core` has NO knowledge of MCP or CLI
2. **Tools and CLI are siblings**: Both depend on core, neither depends on the other
3. **Output formatting is CLI-only**: MCP uses `responses.py`, CLI uses `cli/output.py`
4. **CLI-specific features stay in CLI**: Context tracking, session markers, work mode

### Dependency Graph

```
                    ┌─────────────────┐
                    │  foundry_mcp    │
                    │    .core        │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐       ┌───────────▼───────────┐
    │  foundry_mcp      │       │   foundry_mcp         │
    │    .tools (MCP)   │       │     .cli              │
    └───────────────────┘       └───────────────────────┘
              │                             │
              │                   ┌─────────▼─────────┐
              │                   │ foundry_mcp.cli   │
              │                   │   .commands       │
              │                   └───────────────────┘
              ▼                             ▼
         MCP Server                    Click CLI
```

## Implementation Notes

### Entry Point

```python
# foundry_mcp/cli/main.py
import click
from foundry_mcp.cli.registry import register_all_commands

@click.group()
@click.option('--specs-dir', envvar='SDD_SPECS_DIR', help='Override specs directory')
@click.pass_context
def cli(ctx, specs_dir):
    """SDD CLI - JSON-only output for AI assistants."""
    ctx.ensure_object(dict)
    ctx.obj['specs_dir'] = specs_dir

register_all_commands(cli)

if __name__ == '__main__':
    cli()
```

### pyproject.toml Entry Point

```toml
[project.scripts]
sdd = "foundry_mcp.cli.main:cli"
```

### Command Pattern

```python
# foundry_mcp/cli/commands/task.py
import click
from foundry_mcp.core.task import get_next_task
from foundry_mcp.cli.output import emit, emit_error

@click.group()
def task():
    """Task operations."""
    pass

@task.command('next')
@click.argument('spec_id')
@click.pass_context
def next_task(ctx, spec_id):
    """Find next actionable task."""
    specs_dir = ctx.obj.get('specs_dir')
    try:
        result = get_next_task(spec_id, specs_dir)
        emit({"success": True, "data": result})
    except Exception as e:
        emit_error(str(e))
```

## Alternatives Considered

### Alternative 1: Flat CLI Module

```
foundry_mcp/
├── cli.py              # Single file with all commands
```

**Rejected:** Not scalable with 80+ commands. Hard to maintain.

### Alternative 2: CLI as Separate Package

```
foundry_mcp/            # MCP only
sdd_cli/                # Separate package
```

**Rejected:** Duplicates code, harder to keep in sync, more complex build.

### Alternative 3: CLI Wraps MCP Tools

```python
# CLI calls MCP tools directly
from foundry_mcp.tools.tasks import task_next
result = await task_next(spec_id=spec_id)
```

**Rejected:** MCP tools have async signatures and MCP-specific decorators. CLI should call core directly.

## Consequences

### Positive

- Single package with both MCP and CLI interfaces
- Shared core logic reduces duplication
- CLI can be tested independently of MCP server
- Clear separation of concerns

### Negative

- CLI adds `click` dependency
- CLI-specific features (context tracking) require CLI-only code
- Two surfaces to maintain for each new feature

### Neutral

- Need to ensure core functions work synchronously for CLI
- May need thin async wrappers in MCP tools over sync core functions

## Shared Helper Strategy

The CLI implementation leverages existing core helpers to avoid duplication and ensure consistent behavior across MCP and CLI surfaces.

### Core Helper Modules

| Module | Purpose | CLI Usage |
|--------|---------|-----------|
| `core/responses.py` | Response envelope helpers | Adapt for CLI JSON output |
| `core/validation.py` | Spec validation + auto-fix | Direct use for `sdd validate` |
| `core/security.py` | Input limits and sanitization | Direct use for path/input validation |
| `core/observability.py` | Telemetry and tracing | Required for CLI command tracking (parity with MCP) |
| `core/resilience.py` | Retry and timeout patterns | Use for file operations |
| `core/pagination.py` | Cursor-based pagination | Adapt for CLI list commands |

### Response Adaptation Pattern

MCP tools use `core/responses.py` for structured envelopes:

```python
# MCP tool response
{
    "success": True,
    "data": {"task_id": "task-1-1", "status": "completed"},
    "error": None,
    "meta": {"version": "response-v2", "request_id": "req_abc123"}
}
```

CLI uses the same envelope structure (JSON-only, no mode switching):

```python
# foundry_mcp/cli/output.py
import json
import sys

def emit(data):
    """Emit JSON to stdout. Always JSON, no mode switching."""
    print(json.dumps(data, indent=2, default=str))

def emit_error(message, code="error"):
    """Emit error JSON to stderr and exit with code 1."""
    error = {"success": False, "error": {"code": code, "message": message}}
    print(json.dumps(error, indent=2), file=sys.stderr)
    sys.exit(1)

def emit_success(data, meta=None):
    """Emit success envelope to stdout."""
    response = {"success": True, "data": data, "error": None}
    if meta:
        response["meta"] = meta
    emit(response)
```

### Validation Helper Usage

CLI commands use core validation directly:

```python
# foundry_mcp/cli/commands/validation.py
from foundry_mcp.core.validation import validate_spec, apply_fixes
from foundry_mcp.cli.output import emit_success, emit_error

@validation.command('validate')
@click.argument('spec_id')
@click.pass_context
def validate(ctx, spec_id):
    """Validate a spec file."""
    specs_dir = ctx.obj.get('specs_dir')
    result = validate_spec(spec_id, specs_dir)  # Direct core call
    emit_success(result.to_dict())

@validation.command('fix')
@click.argument('spec_id')
@click.option('--dry-run', is_flag=True)
@click.pass_context
def fix(ctx, spec_id, dry_run):
    """Auto-fix validation issues."""
    specs_dir = ctx.obj.get('specs_dir')
    result = apply_fixes(spec_id, specs_dir, dry_run=dry_run)  # Direct core call
    emit_success(result.to_dict())
```

### Security Helpers

Both MCP and CLI use the same security limits:

```python
# foundry_mcp/core/security.py
MAX_INPUT_SIZE = 10_000_000      # 10MB max file size
MAX_ARRAY_LENGTH = 10_000        # Max items in arrays
MAX_STRING_LENGTH = 1_000_000    # Max string length
MAX_NESTED_DEPTH = 50            # Max JSON nesting
```

CLI uses these for input validation:

```python
# foundry_mcp/cli/commands/spec.py
from foundry_mcp.core.security import MAX_INPUT_SIZE, validate_file_size

@spec.command('create')
@click.argument('name')
@click.option('--template', type=click.Path(exists=True))
def create(name, template):
    if template:
        validate_file_size(template, MAX_INPUT_SIZE)  # Reuse core security
    # ... rest of implementation
```

### Pagination in CLI

List commands adapt core pagination for CLI output:

```python
# foundry_mcp/cli/commands/task.py
from foundry_mcp.core.pagination import paginate_results
from foundry_mcp.cli.output import emit_success

@task.command('list')
@click.argument('spec_id')
@click.option('--limit', default=50)
@click.option('--cursor')
@click.pass_context
def list_tasks(ctx, spec_id, limit, cursor):
    """List tasks with pagination."""
    specs_dir = ctx.obj.get('specs_dir')
    result = paginate_results(
        query_tasks(spec_id, specs_dir),
        cursor=cursor,
        limit=limit
    )
    emit_success(result)
```

### Resilience Patterns

File operations use core retry logic:

```python
# foundry_mcp/cli/commands/spec.py
from foundry_mcp.core.resilience import with_retry, FileOperationError

@spec.command('save')
@click.argument('spec_id')
def save(spec_id):
    """Save spec to disk with retry."""
    try:
        with_retry(
            lambda: write_spec(spec_id),
            max_attempts=3,
            retry_on=(IOError, PermissionError)
        )
    except FileOperationError as e:
        output_error(str(e))
```

### CLI-Only Helpers

Some helpers are CLI-specific and don't belong in core:

| CLI Module | Purpose |
|------------|---------|
| `cli/output.py` | JSON output (emit, emit_error, emit_success) |
| `cli/config.py` | Work mode, session config |
| `cli/context.py` | Token tracking, session markers |

These are NOT shared with MCP tools.

### Helper Selection Criteria

When implementing a CLI command, use this decision tree:

1. **Is the logic transport-agnostic?** → Use `core/*` directly
2. **Is it about response formatting?** → Use `cli/output.py` adapting core envelopes
3. **Is it about input validation/security?** → Use `core/security.py` + `core/validation.py`
4. **Is it about pagination/limits?** → Use `core/pagination.py`
5. **Is it CLI-only (terminal, session)?** → Create in `cli/*`

## Related Documents

- **MCP Best Practices:** See `docs/mcp_best_practices/` for tool design guidance
- **CLI Best Practices:** See `docs/cli_best_practices/` for CLI implementation patterns
