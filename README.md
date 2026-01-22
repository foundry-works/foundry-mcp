# foundry-mcp

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://pypi.org/project/foundry-mcp/)

**Turn AI coding assistants into reliable software engineers with structured specs, progress tracking, and automated review.**

## Table of Contents

- [Why foundry-mcp?](#why-foundry-mcp)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Documentation](#documentation)
- [Scope and Limitations](#scope-and-limitations)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Why foundry-mcp?

**The problem:** AI coding assistants are powerful but unreliable on complex tasks. They lose context mid-feature, skip steps without warning, and deliver inconsistent results across sessions.

**The solution:** foundry-mcp provides the scaffolding to break work into specs, track progress, and verify outputs—so your AI assistant delivers like a professional engineer.

- **No more lost context** — Specs persist state across sessions so the AI picks up where it left off.
- **No more skipped steps** — Task dependencies and blockers ensure nothing gets missed.
- **No more guessing progress** — See exactly what's done, what's blocked, and what's next.
- **No more manual review** — AI review validates implementation against spec requirements.

## Key Features

- **Specs keep AI on track** — Break complex work into phases and tasks the AI can complete without losing context.
- **Progress you can see** — Track what's done, what's blocked, and what's next across multi-session work.
- **AI-powered review** — LLM integration reviews specs, generates PR descriptions, and validates implementation.
- **Works with your tools** — Runs as MCP server (Claude Code, Gemini CLI) or standalone CLI with JSON output.
- **Security built in** — Workspace scoping, API key auth, rate limits, and audit logging ship by default.
- **Discovery-first** — Capabilities declared in a manifest so clients negotiate features automatically.

## Installation

### Prerequisites

- Python 3.10 or higher
- macOS, Linux, or Windows
- MCP-compatible client (e.g., Claude Code)

### Install with uvx (recommended)

```bash
uvx foundry-mcp
```

### Install with pip

```bash
pip install foundry-mcp
```

### Install from source (development)

```bash
git clone https://github.com/tylerburleigh/foundry-mcp.git
cd foundry-mcp
pip install -e ".[test]"
```

## Quick Start

**1. Install the claude-foundry plugin** (from within Claude Code):

```
/plugin marketplace add foundry-works/claude-foundry
/plugin install foundry@claude-foundry
```

Restart Claude Code and trust the repository when prompted.

> **Note:** The plugin automatically registers the MCP server using `uvx` — no separate installation needed.

**2. Run setup:**

```
Please run foundry-setup to configure the workspace.
```

**3. Start building:**

```
I want to add user authentication with JWT tokens.
```

Claude creates a spec with phases, tasks, and verification steps. Ask to implement and it works through tasks in dependency order.

## How It Works

foundry-mcp is the **MCP server** that provides the underlying tools and APIs. The [claude-foundry](https://github.com/foundry-works/claude-foundry) plugin provides the **user-facing skills** that orchestrate workflows.

```
You → Claude Code → claude-foundry plugin → foundry-mcp server
         │                  │                      │
         ▼                  ▼                      ▼
      Natural          Skills like            MCP tools for
      language         foundry-spec,          specs, tasks,
      requests         foundry-implement      reviews, etc.
```

| Component | Role |
|-----------|------|
| **foundry-mcp** | MCP server + CLI providing spec/task/review tools |
| **claude-foundry** | Claude Code plugin providing skills and workflow |

For most users, install both and interact through natural language. The plugin handles tool orchestration automatically.

## Configuration

### API Keys

foundry-mcp uses LLM providers for AI-powered features like spec review, consensus, and deep research. Set the API keys for providers you want to use:

```bash
# AI CLI tools (for AI review, consensus)
export CLAUDE_CODE_OAUTH_TOKEN="..."   # Get via: claude setup-token
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="sk-..."
export CURSOR_API_KEY="key-..."

# Deep research providers (for /foundry-research deep workflow)
export TAVILY_API_KEY="..."
export PERPLEXITY_API_KEY="..."
```

### TOML Configuration (Optional)

Configuration loads in layers (later layers override earlier):

1. **User config** (`~/.foundry-mcp.toml`) - User-wide defaults for API keys, preferred providers
2. **Project config** (`./foundry-mcp.toml`) - Project-specific settings

```bash
# For user-wide settings (API keys, provider preferences)
cp samples/home-foundry-mcp.toml ~/.foundry-mcp.toml

# For project-specific settings
cp samples/foundry-mcp.toml ./foundry-mcp.toml
```

## Advanced Usage

### Direct MCP Configuration (without plugin)

For MCP clients other than Claude Code, or if you prefer manual configuration:

```json
{
  "mcpServers": {
    "foundry-mcp": {
      "command": "uvx",
      "args": ["foundry-mcp"],
      "env": {
        "FOUNDRY_MCP_SPECS_DIR": "/path/to/specs"
      }
    }
  }
}
```

<details>
<summary>Using a pip installation instead?</summary>

```json
{
  "mcpServers": {
    "foundry-mcp": {
      "command": "foundry-mcp",
      "env": {
        "FOUNDRY_MCP_SPECS_DIR": "/path/to/specs"
      }
    }
  }
}
```
</details>

### CLI Usage

All MCP tools are also available via CLI with JSON output:

```bash
# Get next task to work on
python -m foundry_mcp.cli task next --specs-dir ./specs

# Validate a spec
python -m foundry_mcp.cli spec validate my-feature-001

# Create a new spec
python -m foundry_mcp.cli authoring create --name "my-feature" --template detailed
```

### Launch as Standalone MCP Server

```bash
foundry-mcp
```

The server advertises its capabilities, feature flags, and response contract so MCP clients (Claude Code, Gemini CLI, etc.) can connect automatically.

## Documentation

### User guides

| Guide | Description |
|-------|-------------|
| [Quick Start](docs/01-quick-start.md) | Get up and running in 5 minutes |
| [Core Concepts](docs/02-core-concepts.md) | Understand specs, phases, and tasks |
| [Workflow Guide](docs/03-workflow-guide.md) | End-to-end development workflows |
| [CLI Reference](docs/04-cli-command-reference.md) | Complete CLI command documentation |
| [MCP Tool Reference](docs/05-mcp-tool-reference.md) | All MCP tools and their parameters |
| [Configuration](docs/06-configuration.md) | Environment variables and TOML setup |
| [Troubleshooting](docs/07-troubleshooting.md) | Common issues and solutions |

### Concepts

| Guide | Description |
|-------|-------------|
| [SDD Philosophy](docs/concepts/sdd-philosophy.md) | Why spec-driven development matters |
| [Response Envelope](docs/concepts/response-envelope.md) | Standardized response format |
| [Spec Schema](docs/concepts/spec-schema.md) | Spec file structure and fields |
| [LLM Configuration](docs/guides/llm-configuration.md) | Provider setup and fallbacks |

### Developer docs

| Guide | Description |
|-------|-------------|
| [Dev Docs Index](dev_docs/README.md) | Entry point for developer documentation |
| [MCP Best Practices](dev_docs/mcp_best_practices/README.md) | Canonical implementation checklist |
| [Response Schema](dev_docs/codebase_standards/mcp_response_schema.md) | Standardized envelope reference |
| [CLI Output Contract](dev_docs/codebase_standards/cli-output.md) | JSON-first CLI expectations |

## Scope and Limitations

**Best for:**
- Multi-step feature development with AI assistants
- Teams wanting structured handoff between AI and human reviewers
- Projects requiring audit trails and progress visibility

**Not suited for:**
- Quick one-off code changes (use your AI assistant directly)
- Non-software tasks (specs are code-focused)
- Fully autonomous AI agents (foundry assumes human oversight)

## Testing

```bash
pytest                                        # Full suite
pytest tests/integration/test_mcp_smoke.py    # MCP smoke tests
pytest tests/integration/test_mcp_tools.py    # Tool contract coverage
```

- Regression tests keep MCP/CLI adapters aligned across surfaces.
- Golden fixtures (`tests/fixtures/golden`) ensure response envelopes, error semantics, and pagination never regress.
- Freshness checks run alongside core unit and integration suites.

## Contributing

Contributions are welcome! Please read the [MCP Best Practices](dev_docs/mcp_best_practices/README.md) before submitting PRs. All changes should keep specs, docs, code, and fixtures in sync.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built by [Tyler Burleigh](https://github.com/tylerburleigh)** · [Report an Issue](https://github.com/tylerburleigh/foundry-mcp/issues) · [View on GitHub](https://github.com/tylerburleigh/foundry-mcp)
