# foundry-mcp

MCP server for SDD toolkit spec management - query, navigate, and manage specification files through the Model Context Protocol.

## Overview

foundry-mcp provides a comprehensive MCP (Model Context Protocol) server that enables AI assistants to interact with SDD (Spec-Driven Development) specifications. It provides 40+ tools, resources, and prompts for:

- **Spec Management**: List, find, and navigate specifications
- **Task Operations**: Track progress, manage dependencies, prepare task context
- **Validation & Fixes**: Validate specs and auto-fix common issues
- **Journaling**: Track decisions, blockers, and status changes
- **Lifecycle Management**: Move specs through pending → active → completed → archived
- **Code Documentation**: Query codebase docs, trace call graphs, analyze impact
- **Testing**: Run pytest with presets, discover tests, track results
- **Resources & Prompts**: Access specs via URIs, use workflow prompts

## Installation

### Using pip

```bash
pip install foundry-mcp
```

### Using uvx (recommended for Claude Desktop)

```bash
uvx foundry-mcp
```

### From source

```bash
git clone https://github.com/tylerburleigh/foundry-mcp.git
cd foundry-mcp
pip install -e .
```

## Claude Desktop Configuration

Add foundry-mcp to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "foundry-mcp": {
      "command": "uvx",
      "args": ["foundry-mcp"],
      "env": {
        "FOUNDRY_MCP_SPECS_DIR": "/path/to/your/specs"
      }
    }
  }
}
```

Or if installed via pip:

```json
{
  "mcpServers": {
    "foundry-mcp": {
      "command": "foundry-mcp",
      "env": {
        "FOUNDRY_MCP_SPECS_DIR": "/path/to/your/specs"
      }
    }
  }
}
```

## Tools Reference

### Spec Management

| Tool | Description |
|------|-------------|
| `list_specs` | List all specs with optional status filtering |
| `get_spec` | Get summary info about a specification |
| `get_spec_hierarchy` | Get full task hierarchy |
| `get_task` | Get details about a specific task |

### Query Tools

| Tool | Description |
|------|-------------|
| `foundry_query_tasks` | Query tasks by status, type, or parent |

### Task Operations

| Tool | Description |
|------|-------------|
| `foundry_prepare_task` | Prepare task with full context (dependencies, sibling info, phase) |
| `foundry_next_task` | Find next actionable task based on status and dependencies |
| `foundry_task_info` | Get detailed task information |
| `foundry_check_deps` | Check dependency status for a task |
| `foundry_update_status` | Update task status (pending, in_progress, completed, blocked) |
| `foundry_complete_task` | Mark task complete with journal entry |
| `foundry_start_task` | Start working on a task |
| `foundry_progress` | Get spec/phase progress summary |

### Validation Tools

| Tool | Description |
|------|-------------|
| `foundry_validate_spec` | Validate spec and return structured diagnostics |
| `foundry_fix_spec` | Apply auto-fixes with dry-run support |
| `foundry_spec_stats` | Get comprehensive spec statistics |
| `foundry_validate_and_fix` | Combined validation and fix in one operation |

### Journal Tools

| Tool | Description |
|------|-------------|
| `foundry_add_journal` | Add journal entries with optional task association |
| `foundry_get_journal` | Retrieve journal entries with filtering |
| `foundry_mark_blocked` | Mark tasks as blocked with metadata |
| `foundry_unblock` | Unblock tasks and track resolution |
| `foundry_list_blocked` | List all blocked tasks |
| `foundry_unjournaled_tasks` | Find tasks needing journal entries |

### Rendering Tools

| Tool | Description |
|------|-------------|
| `foundry_render_spec` | Render spec to markdown with mode options |
| `foundry_render_progress` | ASCII progress bars for spec and phases |
| `foundry_list_tasks` | Flat task list with filtering |

### Lifecycle Tools

| Tool | Description |
|------|-------------|
| `foundry_move_spec` | Move spec between folders |
| `foundry_activate_spec` | Activate pending spec (pending → active) |
| `foundry_complete_spec` | Mark spec completed (active → completed) |
| `foundry_archive_spec` | Archive spec (completed → archived) |
| `foundry_lifecycle_state` | Get lifecycle state and transition eligibility |
| `foundry_list_specs_by_folder` | List specs organized by folder |

### Documentation Tools

| Tool | Description |
|------|-------------|
| `foundry_find_class` | Find class in codebase documentation |
| `foundry_find_function` | Find function in documentation |
| `foundry_trace_calls` | Trace call graph (callers/callees) with depth |
| `foundry_impact_analysis` | Analyze change impact across codebase |
| `foundry_get_callers` | Get caller functions |
| `foundry_get_callees` | Get callee functions |
| `foundry_docs_stats` | Get documentation statistics |

### Testing Tools

| Tool | Description |
|------|-------------|
| `foundry_run_tests` | Full-featured test execution with presets |
| `foundry_discover_tests` | Test discovery without execution |
| `foundry_test_presets` | List available test presets |
| `foundry_run_quick_tests` | Quick test preset (60s timeout, fail-fast) |
| `foundry_run_unit_tests` | Unit test preset |

## Resources

Access spec data through MCP resources:

| Resource URI | Description |
|--------------|-------------|
| `foundry://specs/` | List all specifications |
| `foundry://specs/{status}/` | List specs by status (active, pending, completed, archived) |
| `foundry://specs/{status}/{spec_id}` | Get specific spec with full hierarchy |
| `foundry://specs/{spec_id}/journal` | Get journal entries for a spec |
| `foundry://templates/` | List available spec templates |
| `foundry://templates/{template_id}` | Get specific template |

### Built-in Templates

- **basic**: Minimal spec with single phase
- **feature**: Standard feature with design, implementation, verification phases
- **bugfix**: Bug investigation, fix, and verification

## Prompts

Workflow prompts for common operations:

| Prompt | Description |
|--------|-------------|
| `start_feature` | Guide new feature setup with template selection |
| `debug_test` | Systematic test debugging workflow |
| `complete_phase` | Phase completion checklist and guidance |
| `review_spec` | Comprehensive spec status review |

## Configuration

foundry-mcp can be configured via environment variables or a TOML config file.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FOUNDRY_MCP_SPECS_DIR` | Path to specs directory | Auto-detected |
| `FOUNDRY_MCP_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `FOUNDRY_MCP_API_KEYS` | Comma-separated API keys for authentication | None |
| `FOUNDRY_MCP_REQUIRE_AUTH` | Require API key authentication | false |

### TOML Configuration

Create a `foundry-mcp.toml` file:

```toml
[workspace]
specs_dir = "/path/to/specs"

[logging]
level = "INFO"
structured = true

[auth]
require_auth = false
api_keys = ["key1", "key2"]

[server]
name = "foundry-mcp"
version = "0.2.0"
```

## Spec Directory Structure

foundry-mcp expects specs organized in status folders:

```
specs/
├── pending/      # New specs awaiting activation
├── active/       # Currently being worked on
├── completed/    # Finished specs
├── archived/     # Historical specs
└── templates/    # Custom spec templates (optional)
```

## Development

```bash
# Clone the repository
git clone https://github.com/tylerburleigh/foundry-mcp.git
cd foundry-mcp

# Install in development mode
pip install -e .

# Run tests
pytest

# Run the server
foundry-mcp
```

## License

MIT
