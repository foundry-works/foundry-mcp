# Testing Runbook - foundry-mcp

A comprehensive guide for developers to test the foundry-mcp MCP server.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Test Directory Structure](#test-directory-structure)
- [Running Tests with Pytest](#running-tests-with-pytest)
- [Manual MCP Testing](#manual-mcp-testing)
- [End-to-End Workflow Testing](#end-to-end-workflow-testing)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Prerequisites

### Requirements

- Python 3.10+
- pytest

### Installation

```bash
# Install package in development mode
pip install -e .

# Install test dependencies
pip install pytest

# Verify installation
python -c "import foundry_mcp; print('foundry-mcp installed')"
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `FOUNDRY_MCP_SPECS_DIR` | Path to specs directory | `/path/to/specs` |
| `FOUNDRY_MCP_WORKSPACE_ROOTS` | Comma-separated workspace roots | `/path/one,/path/two` |
| `FOUNDRY_MCP_JOURNALS_PATH` | Path to journals directory | `/path/to/journals` |
| `FOUNDRY_MCP_LOG_LEVEL` | Logging level | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `FOUNDRY_MCP_CONFIG_FILE` | Path to TOML config file | `./foundry-mcp.toml` |
| `FOUNDRY_MCP_API_KEYS` | Comma-separated API keys (optional) | `key1,key2` |
| `FOUNDRY_MCP_REQUIRE_AUTH` | Require API key auth | `true` or `false` |

### TOML Configuration

You can also use a `foundry-mcp.toml` file:

```toml
[workspace]
specs_dir = "/path/to/specs"
journals_path = "/path/to/journals"

[logging]
level = "INFO"
structured = true

[auth]
require_auth = false
```

---

## Test Directory Structure

```
tests/
├── unit/                    # Unit tests for core modules
│   ├── test_spec.py         # Spec operations
│   ├── test_task.py         # Task operations
│   ├── test_validation.py   # Validation logic
│   ├── test_mutations.py    # Mutation operations
│   ├── test_authoring.py    # Authoring tools
│   └── ...                  # Additional unit tests
├── integration/             # Integration tests
│   ├── test_mcp_smoke.py    # Server creation/tool registration
│   ├── test_mcp_tools.py    # Tool schemas, resources, prompts
│   ├── test_llm_tools.py    # LLM-powered tool tests
│   ├── test_llm_review.py   # Review tool tests
│   └── test_fallback_integration.py
├── fixtures/                # Test fixtures and sample specs
│   └── golden/              # Golden fixture files
└── parity/                  # CLI parity tests (internal)
    ├── conftest.py          # Shared fixtures
    └── harness/             # Test harness utilities
```

---

## Running Tests with Pytest

### Basic Commands

```bash
# Run all tests
pytest

# Run all tests (explicit path)
pytest tests/

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_core/test_spec.py

# Run specific test class
pytest tests/unit/test_core/test_spec.py::TestLoadSpec

# Run specific test method
pytest tests/unit/test_core/test_spec.py::TestLoadSpec::test_load_spec_success

# Run tests in a directory
pytest tests/unit/

# Run tests matching a pattern
pytest -k "test_spec"
```

### Using Pytest Markers

The project defines custom markers in `pytest.ini`:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Exclude integration tests
pytest -m "not integration"

# Combine markers
pytest -m "unit and not slow"
```

### Built-in Test Presets

The project includes preset configurations in `src/foundry_mcp/core/testing.py`:

| Preset | Timeout | Fail Fast | Markers | Use Case |
|--------|---------|-----------|---------|----------|
| `quick` | 60s | Yes | `not slow` | Fast feedback during development |
| `full` | 300s | No | None | Complete test run |
| `unit` | 120s | No | `unit` | Unit tests only |
| `integration` | 300s | No | `integration` | Integration tests only |
| `smoke` | 30s | Yes | `smoke` | Quick verification |

**Preset equivalent commands:**

```bash
# Quick preset - fast development feedback
pytest -m "not slow" -x --timeout=60

# Full preset - complete test run
pytest -v --timeout=300

# Unit preset
pytest -m unit -v --timeout=120

# Integration preset
pytest -m integration -v --timeout=300

# Smoke preset
pytest -m smoke -x --timeout=30
```

### Debugging Tests

```bash
# Show print statements (disable output capture)
pytest -s

# Show full traceback
pytest --tb=long

# Stop on first failure
pytest -x

# Run only last failed tests
pytest --lf

# Show local variables in traceback
pytest -l

# Run failed tests first
pytest --ff

# Generate coverage report
pytest --cov=foundry_mcp --cov-report=html
```

---

## Manual MCP Testing

### Starting the Server

```bash
# Direct run (after pip install)
foundry-mcp

# Run from source
python -m foundry_mcp.server

# With environment configuration
FOUNDRY_MCP_SPECS_DIR=/path/to/specs foundry-mcp

# With TOML config
FOUNDRY_MCP_CONFIG_FILE=./foundry-mcp.toml foundry-mcp

# With debug logging
FOUNDRY_MCP_LOG_LEVEL=DEBUG foundry-mcp
```

### MCP Dev Mode

Use the MCP CLI to test interactively:

```bash
# Install MCP CLI (if not installed)
pip install mcp

# Run in development mode
mcp dev foundry-mcp

# Or with uvx
uvx --from mcp dev foundry-mcp
```

### Claude Desktop Integration

**Configuration file locations:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Example configuration (uvx):**

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

**Example configuration (pip installed):**

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

### Testing Tools Manually

Once connected via Claude Desktop or MCP dev mode, test these tools:

**Task Tool (`task`)**
- `task(action="next")` - Find next actionable task
- `task(action="prepare")` - Get task context and dependencies
- `task(action="start")` - Mark task as in_progress
- `task(action="complete")` - Mark task as completed
- `task(action="info")` - Get detailed task information
- `task(action="progress")` - Get spec progress summary
- `task(action="block")` / `task(action="unblock")` - Manage blocked tasks

**Spec Tool (`spec`)**
- `spec(action="list")` / `spec(action="find")` - Discover specs
- `spec(action="validate")` - Validate spec structure
- `spec(action="fix")` - Auto-fix validation issues
- `spec(action="stats")` - Get spec statistics

**Lifecycle Tool (`lifecycle`)**
- `lifecycle(action="activate")` - Move spec from pending to active
- `lifecycle(action="complete")` - Move spec from active to completed
- `lifecycle(action="archive")` - Move spec to archived
- `lifecycle(action="state")` - Get current lifecycle state

**Journal Tool (`journal`)**
- `journal(action="add")` - Add journal entry
- `journal(action="list")` - Get journal entries
- `journal(action="list-unjournaled")` - List completed tasks needing journals

**Test Tool (`test`)**
- `test(action="run", preset="full")` - Run pytest suite
- `test(action="discover")` - Discover available tests

### Testing Resources

Resources are accessed via `foundry://` URIs:

```
foundry://specs/                    # List all specs
foundry://specs/active/             # List active specs
foundry://specs/pending/            # List pending specs
foundry://specs/completed/          # List completed specs
foundry://specs/archived/           # List archived specs
foundry://specs/{status}/{spec_id}  # Get specific spec
foundry://specs/{spec_id}/journal   # Get spec journal
foundry://templates/                # List templates
foundry://templates/basic           # Basic template
foundry://templates/feature         # Feature template
foundry://templates/bugfix          # Bugfix template
```

### Testing Prompts

Workflow prompts available:
- `start_feature` - New feature setup workflow
- `debug_test` - Test debugging workflow
- `complete_phase` - Phase completion checklist
- `review_spec` - Spec status review

---

## End-to-End Workflow Testing

Test a complete SDD workflow manually:

### Step-by-Step Workflow

1. **Create spec from template**
   - Use `foundry://templates/feature` to get template
   - Create spec file in `specs/pending/`

2. **Activate spec**
   - `lifecycle(action="activate", spec_id="your-spec-id")`
   - Verify spec moves from `pending/` to `active/`

3. **Find next task**
   - `task(action="next", spec_id="your-spec-id")`
   - Returns first actionable task with dependencies met

4. **Prepare task context**
   - `task(action="prepare", spec_id="your-spec-id", task_id="1.1")`
   - Returns task details, dependencies, and context

5. **Start task**
   - `task(action="start", spec_id="your-spec-id", task_id="1.1")`
   - Task status changes to `in_progress`

6. **Complete task**
   - `task(action="complete", spec_id="your-spec-id", task_id="1.1")`
   - Task status changes to `completed`

7. **Check progress**
   - `task(action="progress", spec_id="your-spec-id")`
   - Returns completion percentage and phase summary

8. **Complete spec**
   - `lifecycle(action="complete", spec_id="your-spec-id")`
   - Spec moves from `active/` to `completed/`

9. **Archive spec**
   - `lifecycle(action="archive", spec_id="your-spec-id")`
   - Spec moves from `completed/` to `archived/`

### Verification Checklist

- [ ] Server starts without errors
- [ ] Unified tools are registered (17 when `unified_manifest` is enabled)
- [ ] Resources are accessible via `foundry://` URIs
- [ ] Prompts expand correctly
- [ ] Lifecycle transitions work (pending -> active -> completed -> archived)
- [ ] Journal entries are recorded
- [ ] Progress calculations are accurate
- [ ] Validation catches spec issues
- [ ] Auto-fix resolves fixable issues

---

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: foundry_mcp` | Package not installed | `pip install -e .` |
| `No specs directory found` | Missing or misconfigured specs path | Set `FOUNDRY_MCP_SPECS_DIR` |
| `pytest not found` | pytest not installed | `pip install pytest` |
| `Spec not found: {spec_id}` | Invalid spec ID or wrong folder | Check spec exists in status folders |
| `Invalid spec structure` | Malformed JSON | Run `spec(action="validate")` |
| `Dependency not met` | Task has incomplete dependencies | Complete dependency tasks first |

### Debugging Server Issues

```bash
# Check config loading
python -c "from foundry_mcp.config import get_config; print(get_config())"

# Verify specs directory structure
ls -la /path/to/specs/{pending,active,completed,archived}

# Check for missing dependencies
pip list | grep -E "(fastmcp|mcp)"

# Run with debug logging
FOUNDRY_MCP_LOG_LEVEL=DEBUG foundry-mcp
```

### Test Discovery Issues

```bash
# Check pytest configuration
cat pytest.ini

# Discover tests manually
pytest --collect-only

# Check for import errors
python -c "from foundry_mcp.server import create_server"

# Run specific test file to check imports
pytest tests/unit/test_core/test_spec.py --collect-only
```

---

## Best Practices

1. **Use fixtures for temporary specs**
   - Always use `tempfile.TemporaryDirectory` for test isolation
   - Create proper folder structure: `pending/`, `active/`, `completed/`, `archived/`

2. **Follow the fixture pattern:**
   ```python
   @pytest.fixture
   def temp_specs_dir():
       with tempfile.TemporaryDirectory() as tmpdir:
           specs_dir = Path(tmpdir) / "specs"
           for status in ["pending", "active", "completed", "archived"]:
               (specs_dir / status).mkdir(parents=True)
           yield specs_dir
   ```

3. **Use presets appropriately:**
   - `quick` preset during active development
   - `full` preset before committing
   - `smoke` preset for CI quick checks

4. **Mark slow tests:**
   ```python
   @pytest.mark.slow
   def test_expensive_operation():
       ...
   ```

5. **Test in isolation:**
   - Each test should be independent
   - Don't rely on test execution order
   - Clean up any created resources

6. **Run integration tests separately:**
   ```bash
   # Quick unit test feedback
   pytest -m "not integration" -x

   # Full integration run
   pytest -m integration -v
   ```
