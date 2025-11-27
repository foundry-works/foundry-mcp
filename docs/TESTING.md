# Testing Runbook - foundry-mcp

A comprehensive guide for developers to test the foundry-mcp MCP server.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Test Directory Structure](#test-directory-structure)
- [Running Tests with Pytest](#running-tests-with-pytest)
- [Parity Testing](#parity-testing)
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
│   └── test_core/          # Core module tests
│       ├── test_spec.py    # Spec operations
│       ├── test_task.py    # Task operations
│       ├── test_validation.py
│       ├── test_journal.py
│       ├── test_lifecycle.py
│       └── test_rendering.py
├── integration/            # Integration tests
│   ├── test_mcp_smoke.py   # Server creation/tool registration
│   ├── test_mcp_tools.py   # Tool schemas, resources, prompts
│   ├── test_notifications_sampling.py
│   ├── test_prepare_task_cli.py
│   └── test_fallback_integration.py
├── parity/                 # Parity tests (foundry-mcp vs sdd-toolkit)
│   ├── conftest.py         # Shared fixtures
│   ├── test_lifecycle.py   # Lifecycle operation parity
│   ├── test_spec_operations.py
│   ├── test_task_operations.py
│   ├── test_validation.py
│   ├── test_progress_tracking.py
│   ├── fixtures/           # Test spec fixtures
│   └── harness/            # Adapter and comparator utilities
├── skills/                 # Skill-specific tests
│   └── llm_doc_gen/
├── doc_query/              # Documentation query tests
├── sdd_next/               # SDD next task preparation tests
└── fixtures/               # Test fixtures
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

## Parity Testing

Parity tests ensure that foundry-mcp produces equivalent results to sdd-toolkit for shared operations. This helps maintain compatibility between the Python library and the CLI tool.

### Parity Testing Overview

Parity tests compare the output of foundry-mcp core functions against sdd-toolkit CLI commands. Both systems operate on the same spec format, so their results should be functionally equivalent.

**Key concepts:**
- **Adapters**: Thin wrappers that normalize API calls for each system
- **Comparators**: Utilities that compare normalized results
- **Fixtures**: Sample specs used for testing
- **Test categories**: Spec operations, task operations, lifecycle, validation, progress tracking

**Test directory structure:**
```
tests/parity/
├── conftest.py           # Shared pytest fixtures
├── test_spec_operations.py
├── test_task_operations.py
├── test_lifecycle.py
├── test_validation.py
├── test_progress_tracking.py
├── fixtures/
│   └── sample_specs/     # Test spec fixtures
│       ├── simple_spec.json
│       ├── completed_spec.json
│       ├── multi_phase_spec.json
│       ├── with_dependencies.json
│       └── with_blockers.json
└── harness/
    ├── base.py           # Base adapter interface
    ├── foundry_adapter.py # Foundry-mcp adapter
    ├── sdd_adapter.py    # SDD-toolkit CLI adapter
    ├── comparators.py    # Result comparison utilities
    ├── normalizers.py    # Response normalization
    └── fixture_manager.py # Fixture setup utilities
```

### Running Parity Tests

```bash
# Run all parity tests
pytest tests/parity/ -v -m parity

# Run parity tests for specific category
pytest tests/parity/test_lifecycle.py -v -m parity
pytest tests/parity/test_spec_operations.py -v -m parity
pytest tests/parity/test_task_operations.py -v -m parity
pytest tests/parity/test_validation.py -v -m parity
pytest tests/parity/test_progress_tracking.py -v -m parity

# Run only parity-marked tests (excludes standalone foundry tests)
pytest -m parity

# Run parity tests with detailed output
pytest tests/parity/ -v --tb=short -m parity

# Run parity tests excluding slow ones
pytest tests/parity/ -v -m "parity and not slow"
```

**Quick verification:**
```bash
# Check parity marker is registered
pytest --markers | grep parity

# Discover parity tests
pytest tests/parity/ --collect-only -m parity
```

### Adding New Parity Tests

#### Step 1: Create or reuse a fixture

Fixtures are sample specs in `tests/parity/fixtures/sample_specs/`. Use existing ones or create new ones:

```json
{
  "spec_id": "parity-test-example",
  "title": "Example Spec",
  "metadata": {
    "title": "Example Spec",
    "description": "For parity testing",
    "version": "1.0.0"
  },
  "hierarchy": {
    "spec-root": {
      "type": "spec",
      "title": "Example Spec",
      "status": "in_progress",
      "children": ["phase-1"]
    },
    "phase-1": {
      "type": "phase",
      "title": "Phase 1",
      "status": "in_progress",
      "parent": "spec-root",
      "children": ["task-1-1"]
    },
    "task-1-1": {
      "type": "task",
      "title": "First task",
      "status": "pending",
      "parent": "phase-1",
      "children": []
    }
  }
}
```

#### Step 2: Write the parity test

```python
import pytest
from .harness.comparators import ResultComparator
from .harness.fixture_manager import FixtureManager
from .harness.foundry_adapter import FoundryMcpAdapter
from .harness.sdd_adapter import SddToolkitAdapter


class TestMyOperation:
    """Parity tests for my operation."""

    @pytest.mark.parity
    def test_my_operation_parity(self, test_dir):
        """Test that my_operation produces equivalent results."""
        # Setup isolated copies for each system
        foundry_root = test_dir / "foundry"
        foundry_fixture = FixtureManager(foundry_root)
        foundry_fixture.setup("simple_spec", status="active")
        foundry = FoundryMcpAdapter(foundry_root / "specs")

        sdd_root = test_dir / "sdd"
        sdd_fixture = FixtureManager(sdd_root)
        sdd_fixture.setup("simple_spec", status="active")
        sdd = SddToolkitAdapter(sdd_root / "specs")

        # Execute operation on both systems
        foundry_result = foundry.my_operation("parity-test-simple")
        sdd_result = sdd.my_operation("parity-test-simple")

        # Compare results
        ResultComparator.assert_success(
            foundry_result, sdd_result, "my_operation"
        )

    @pytest.mark.parity
    def test_my_operation_error_parity(self, test_dir):
        """Test that error handling is equivalent."""
        # Setup fixtures...

        # Execute with invalid input
        foundry_result = foundry.my_operation("nonexistent")
        sdd_result = sdd.my_operation("nonexistent")

        # Both should error
        ResultComparator.assert_both_error(
            foundry_result, sdd_result, "my_operation(nonexistent)"
        )
```

#### Step 3: Add adapter methods if needed

For new operations, add methods to both adapters:

**foundry_adapter.py:**
```python
def my_operation(self, spec_id: str) -> Dict[str, Any]:
    """Perform my operation."""
    try:
        spec_data = load_spec(spec_id, self.specs_dir)
        if spec_data is None:
            return {"success": False, "error": f"Spec not found: {spec_id}"}

        result = core_my_operation(spec_data)
        return {
            "success": True,
            "result_field": result.value,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**sdd_adapter.py:**
```python
def my_operation(self, spec_id: str) -> Dict[str, Any]:
    """Perform my operation via CLI."""
    return self._run_sdd("my-operation", spec_id)
```

#### Step 4: Use appropriate fixtures

| Fixture | Use case |
|---------|----------|
| `both_adapters` | Read-only tests on same data |
| `isolated_adapters` | Tests that modify state |
| `fixture_manager` | Custom fixture setup |
| `test_dir` | Base temporary directory |

### Troubleshooting Parity Failures

#### Common failure patterns

| Error Pattern | Cause | Solution |
|---------------|-------|----------|
| `'DataclassType' object has no attribute 'get'` | Foundry returns dataclass, not dict | Convert dataclass to dict in adapter |
| `Key mismatch for 'field'` | Different response keys | Normalize keys in adapter response |
| `sdd-toolkit failed: unrecognized arguments` | CLI doesn't support that flag | Update sdd_adapter to use correct CLI syntax |
| `Spec not found` | Fixture not set up correctly | Check FixtureManager.setup() call |
| `'list' object has no attribute 'get'` | Result is list, expected dict | Wrap list in success response dict |

#### Debugging strategies

```bash
# Run single failing test with full output
pytest tests/parity/test_spec_operations.py::TestListSpecs::test_list_all_specs_parity -v --tb=long -s

# Check adapter output directly
python -c "
from pathlib import Path
import tempfile
from tests.parity.harness.fixture_manager import FixtureManager
from tests.parity.harness.foundry_adapter import FoundryMcpAdapter

test_dir = Path(tempfile.mkdtemp())
fixture = FixtureManager(test_dir)
fixture.setup('simple_spec', status='active')
adapter = FoundryMcpAdapter(test_dir / 'specs')
result = adapter.my_operation('parity-test-simple')
print(result)
"

# Compare CLI output directly
sdd --json --path /tmp/test my-operation parity-test-simple
```

#### Known limitations

Some operations have expected differences:

- **CLI argument differences**: sdd-toolkit may not support all flags (e.g., `--type` for get-journal)
- **Response format variations**: Some CLI commands return lists directly vs. wrapped dicts
- **Completion requirements**: sdd-toolkit complete-spec requires all tasks done; foundry-mcp can force

Mark tests as expected failures for known issues:
```python
@pytest.mark.parity
@pytest.mark.xfail(reason="CLI doesn't support --type argument")
def test_get_journal_by_type_parity(self, both_adapters):
    ...
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

**Task Management Tools:**
- `task-next` - Find next actionable task
- `task-prepare` - Get task context and dependencies
- `task-start` - Mark task as in_progress
- `task-complete` - Mark task as completed
- `task-info` - Get detailed task information
- `task-progress` - Get spec progress summary

**Validation Tools:**
- `spec-validate` - Validate spec structure
- `spec-fix` - Auto-fix validation issues
- `spec-stats` - Get spec statistics

**Lifecycle Tools:**
- `spec-lifecycle-activate` - Move spec from pending to active
- `spec-lifecycle-complete` - Move spec from active to completed
- `spec-lifecycle-archive` - Move spec to archived
- `spec-lifecycle-state` - Get current lifecycle state

**Journal Tools:**
- `journal-add` - Add journal entry
- `journal-list` - Get journal entries
- `task-block` - Mark task as blocked
- `task-unblock` - Remove blocked status

**Testing Tools:**
- `test-run` - Run tests with preset or custom config
- `test-discover` - Discover available tests
- `test-presets` - List available test presets

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
   - `spec-lifecycle-activate(spec_id="your-spec-id")`
   - Verify spec moves from `pending/` to `active/`

3. **Find next task**
   - `task-next(spec_id="your-spec-id")`
   - Returns first actionable task with dependencies met

4. **Prepare task context**
   - `task-prepare(spec_id="your-spec-id", task_id="1.1")`
   - Returns task details, dependencies, and context

5. **Start task**
   - `task-start(spec_id="your-spec-id", task_id="1.1")`
   - Task status changes to `in_progress`

6. **Complete task**
   - `task-complete(spec_id="your-spec-id", task_id="1.1")`
   - Task status changes to `completed`

7. **Check progress**
   - `task-progress(spec_id="your-spec-id")`
   - Returns completion percentage and phase summary

8. **Complete spec**
   - `spec-lifecycle-complete(spec_id="your-spec-id")`
   - Spec moves from `active/` to `completed/`

9. **Archive spec**
   - `spec-lifecycle-archive(spec_id="your-spec-id")`
   - Spec moves from `completed/` to `archived/`

### Verification Checklist

- [ ] Server starts without errors
- [ ] 40+ tools are registered
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
| `Invalid spec structure` | Malformed JSON | Run `spec-validate` |
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
