# foundry-mcp

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://pypi.org/project/foundry-mcp/)

**An MCP server that brings spec-driven development to your AI assistant.**

Query, navigate, and manage specification files through the Model Context Protocol. Build features methodically with structured tasks, automated validation, and intelligent code analysis.

## Why foundry-mcp?

- 🎯 **Structured Development** — Break down features into trackable tasks with dependencies, phases, and verification steps
- 🤖 **AI-Native** — 40+ MCP tools designed for AI assistants to query specs, track progress, and validate work
- 🔍 **Code Intelligence** — Trace call graphs, analyze impact, and search codebase documentation
- ✅ **Built-in Quality** — Auto-validation, fidelity reviews, and comprehensive test integration
- 📝 **Full Traceability** — Journal decisions, track blockers, and document everything

## Quick Start

### Prerequisites

- Python 3.10 or higher
- An MCP-compatible client (Claude Desktop, Claude Code, etc.)

### Installation

**Option 1: Using uvx (recommended for Claude Desktop)**
```bash
uvx foundry-mcp
```

**Option 2: Using pip**
```bash
pip install foundry-mcp
```

**Option 3: From source**
```bash
git clone https://github.com/tylerburleigh/foundry-mcp.git
cd foundry-mcp
pip install -e .
```

### Claude Desktop Setup

Add foundry-mcp to your Claude Desktop configuration:

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

<details>
<summary>Using pip installation instead?</summary>

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
</details>

## Features

### 📋 Spec Management

Create and manage specifications through their full lifecycle:

```
specs/
├── pending/      # New specs awaiting activation
├── active/       # Currently being worked on
├── completed/    # Finished specs
└── archived/     # Historical reference
```

| Tool | What it does |
|------|--------------|
| `spec-create` | Scaffold a new specification from templates |
| `spec-validate` | Check structure and return diagnostics |
| `spec-fix` | Auto-fix common issues with dry-run support |
| `spec-lifecycle-*` | Move specs between status folders |

### ✅ Task Operations

Track progress from start to finish:

| Tool | What it does |
|------|--------------|
| `task-next` | Find the next actionable task |
| `task-prepare` | Get full context (dependencies, siblings, phase) |
| `task-start` | Mark a task as in-progress |
| `task-complete` | Mark complete with journal entry |
| `task-block` / `task-unblock` | Manage blockers with metadata |

### 🤖 LLM-Powered Analysis

Intelligent features that gracefully degrade when LLM is unavailable:

| Tool | What it does |
|------|--------------|
| `spec-review` | Get AI-powered improvement suggestions |
| `spec-review-fidelity` | Verify implementation matches spec |
| `spec-doc-llm` | Generate comprehensive documentation |
| `pr-create-with-spec` | Create PRs with AI-enhanced descriptions |

### 🔍 Code Intelligence

Query your codebase documentation:

| Tool | What it does |
|------|--------------|
| `code-find-class` / `code-find-function` | Search definitions |
| `code-trace-calls` | Trace caller/callee relationships |
| `code-impact-analysis` | Analyze change impact |
| `doc-stats` | Get documentation statistics |

### 🧪 Testing Integration

Run and discover tests with pytest presets:

| Tool | What it does |
|------|--------------|
| `test-run` | Execute tests with configurable options |
| `test-run-quick` | Fast tests with fail-fast enabled |
| `test-discover` | Find tests without running them |
| `test-presets` | List available preset configurations |

### 📊 Resources & Prompts

Access data through MCP resources and use workflow prompts:

**Resources:**
- `foundry://specs/` — List all specifications
- `foundry://specs/{status}/{spec_id}` — Get specific spec with hierarchy
- `foundry://templates/` — List available templates

**Prompts:**
- `start_feature` — Guide new feature setup
- `debug_test` — Systematic test debugging
- `complete_phase` — Phase completion checklist

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FOUNDRY_MCP_SPECS_DIR` | Path to specs directory | Auto-detected |
| `FOUNDRY_MCP_LOG_LEVEL` | Logging level | INFO |
| `FOUNDRY_MCP_WORKFLOW_MODE` | Execution mode: `single`, `autonomous`, `batch` | single |

### TOML Configuration

Create `foundry-mcp.toml` for advanced configuration:

```toml
[workspace]
specs_dir = "/path/to/specs"

[logging]
level = "INFO"
structured = true

[workflow]
mode = "single"
auto_validate = true
journal_enabled = true

[llm]
provider = "openai"        # or "anthropic", "local"
model = "gpt-4"            # optional
timeout = 30
```

### LLM Provider Setup

foundry-mcp supports multiple LLM providers for AI-powered features:

| Provider | API Key Variable | Models |
|----------|-----------------|--------|
| OpenAI | `OPENAI_API_KEY` | gpt-4.1, gpt-5.1-codex |
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-5, claude-opus-4-5 |
| Local (Ollama) | None required | Any Ollama model |

LLM features degrade gracefully — specs work fine without an LLM configured.

## Documentation

| Guide | Description |
|-------|-------------|
| [SDD Philosophy](docs/concepts/sdd-philosophy.md) | What is spec-driven development and why use it |
| [Development Guide](docs/guides/development-guide.md) | Setup, architecture, and contributing |
| [Testing Guide](docs/guides/testing.md) | Running and debugging tests |
| [LLM Configuration](docs/guides/llm-configuration.md) | Setting up LLM providers |
| [MCP Best Practices](docs/mcp_best_practices/README.md) | Industry patterns for reliable MCP tools |
| [Response Schema](docs/codebase_standards/mcp_response_schema.md) | Canonical response contract |
| [Tool Naming](docs/codebase_standards/naming-conventions.md) | Naming conventions for MCP operations |

## Development

```bash
# Clone and install
git clone https://github.com/tylerburleigh/foundry-mcp.git
cd foundry-mcp
pip install -e ".[test]"

# Run tests
pytest

# Run the server
foundry-mcp
```

### Project Status

foundry-mcp is in **alpha** development. APIs may change between versions. It's suitable for early adopters building spec-driven workflows. Feedback and contributions are welcome!

## Contributing

Contributions are welcome! Please read the [MCP Best Practices](docs/mcp_best_practices/README.md) before submitting PRs to understand the response contract and testing expectations.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built by [Tyler Burleigh](https://github.com/tylerburleigh)** · [Report an Issue](https://github.com/tylerburleigh/foundry-mcp/issues) · [View on GitHub](https://github.com/tylerburleigh/foundry-mcp)
