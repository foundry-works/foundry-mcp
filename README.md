# foundry-mcp

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://pypi.org/project/foundry-mcp/)

**An MCP server and native CLI that bring spec-driven development to your AI assistant.**

foundry-mcp packages the spec lifecycle, a single CLI/service layer, and MCP adapters described in the completed specs under `specs/completed/`. Every MCP response uses the standardized `response-v2` envelope, the CLI shares the same service layer, and feature-flagged tool suites cover environment setup, authoring, validation, LLM review, and automated testing.

## 🚀 Why foundry-mcp?

- **Single service layer for CLI + MCP** — The completed CLI re-implementation and subprocess elimination specs ensure the CLI and MCP tools share contracts, observability, and feature flags.
- **Spec lifecycle automation** — Tools manage creation, validation, lifecycle transitions, blockers, and journaling with cursor-based pagination and dependency tracking.
- **Quality gates & behavioral testing** — A dedicated regression harness keeps foundry-mcp aligned with the legacy claude-sdd-toolkit CLI while integration/unit/property suites guard regressions.
- **LLM-ready workflows** — Provider abstractions, prompt shielding, and graceful fallbacks power AI review, documentation, and PR creation workflows when LLM access is available.
- **Security & governance baked in** — API keys, workspace scoping, rate limiting, structured logging, and audit trails are enforced before business logic as mandated by the MCP best-practices remediation spec.
- **Discovery-first design** — Capabilities are declared through `mcp/capabilities_manifest.json` so clients can negotiate response contracts, feature flags, and tool availability.

## 📦 Installation

### Pre-requisites

- Python 3.10 or higher
- macOS, Linux, or Windows
- MCP-compatible client (e.g., Claude Code)

### Quick install

#### Run instantly with `uvx`

```bash
uvx foundry-mcp
```

#### Install from PyPI with `pip`

```bash
pip install foundry-mcp
```

#### Install from source (development)

```bash
git clone https://github.com/tylerburleigh/foundry-mcp.git
cd foundry-mcp
pip install -e ".[test]"
```

#### Launch the native SDD CLI

```bash
python -m foundry_mcp.cli --help
```

## 📅 Release cadence & support channels

- The project currently ships **alpha** releases after each spec milestone; see [CHANGELOG.md](CHANGELOG.md) for the latest tagged version.
- PyPI publishes semantic versions that align with the spec roadmap (`response_contract_v2`, feature flags, test suites, etc.).
- MCP capabilities expose rollout state so clients can opt-in to new contracts (for example `response_contract=v2`).

## 📋 Key features

### Spec lifecycle & governance

```
specs/
├── pending/      # New specs awaiting activation
├── active/       # Currently being worked on
├── completed/    # Finished specs (automatically journaled)
└── archived/     # Historical reference
```

- Discover and validate specs via `spec(action=...)`.
- Transition spec folders/states via `lifecycle(action=...)`.
- Automatically journal decisions, blockers, and dependency updates with audit metadata.

### Task operations & execution

- `task(action=next|prepare|start|complete|...)` and blocker flows expose the full dependency graph.
- `plan(action=create|list|review)` supports lightweight planning and review flows.
- Notifications and sampling channels surface phase completions to MCP clients.

### Code, docs, and testing intelligence

- Code navigation tools via `code(action=...)` support symbol lookup and call-graph tracing.
- Testing tools via `test(action=run|discover, preset=quick|unit|full)` run pytest presets with structured output.
- Shared adapters mirror claude-sdd-toolkit behavior and integrate with the regression testing harness.

### LLM-powered workflows

- Configurable provider abstraction with OpenAI, Anthropic, and local backends (Ollama, etc.) plus prompt shielding and observability hooks.
- AI-enhanced review via `review(action=spec|fidelity|parse-feedback)` and PR helpers degrade gracefully when no LLM is configured.
- Timeouts, retries, and circuit breakers follow the resilience patterns from the remediation specs.

### CLI + MCP integration

- Run `foundry-mcp` as an MCP server or `python -m foundry_mcp.cli` for the JSON-first CLI.
- Both surfaces share response helpers, validation, feature flags, and discovery metadata so you can switch between automated MCP clients and terminal workflows without drift.

### Resources & prompts

- `foundry://specs/` resources expose full spec hierarchies, journals, and templates for AI assistants.
- Workflow prompts (`start_feature`, `debug_test`, `complete_phase`, etc.) guide SDD operations end-to-end.

## 🔐 Access & security

- Workspace roots are scoped via configuration to prevent directory escape.
- Optional API keys (`FOUNDRY_MCP_API_KEYS`) or tenant TOML overrides enforce authentication before any tool runs.
- Rate limits and concurrency budgets are declared in the capabilities manifest and enforced server-side with structured audit logs.
- Sensitive data redaction, prompt shielding, and validation helpers protect against prompt injection or oversized payloads.

## 🧾 Response contract & discovery

All MCP tools emit the standardized envelope defined in `docs/codebase_standards/mcp_response_schema.md`:

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "meta": {
    "version": "response-v2",
    "pagination": { ... },
    "warnings": []
  }
}
```

- `success`, `data`, `error`, and `meta` are always present so clients never guess at output shape.
- `response_contract_v2` is feature-flagged; clients advertise support via capability negotiation.
- `mcp/capabilities_manifest.json` advertises the 17 unified tools (plus feature flags like `unified_manifest`).

**Legacy → unified mapping (examples)**

| Legacy tool | Unified call |
|---|---|
| Legacy Tool (Removed) | Unified Equivalent |
|----------------------|--------------------|
| `task-next` | `task(action="next")` |
| `spec-validate` | `spec(action="validate")` |
| `test-run` | `test(action="run", preset="full")` |
| `tool-list` | `server(action="tools")` |
| `get-server-context` | `server(action="context")` |

## ⚙️ Configuration

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FOUNDRY_MCP_SPECS_DIR` | Path to specs directory | Auto-detected from workspace |
| `FOUNDRY_MCP_LOG_LEVEL` | Logging level (DEBUG, INFO, etc.) | `INFO` |
| `FOUNDRY_MCP_WORKFLOW_MODE` | Execution mode: `single`, `autonomous`, `batch` | `single` |
| `FOUNDRY_MCP_API_KEYS` | Comma-separated API keys required for tool access | Disabled |
| `FOUNDRY_MCP_FEATURE_FLAGS` | Additional feature flags to enable (e.g., `planning_tools`) | Based on spec rollout |
| `FOUNDRY_MCP_RESPONSE_CONTRACT` | Force response contract version (`v2`) | Auto-negotiated |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | LLM provider credentials | Not set |

### TOML configuration

Create `foundry-mcp.toml` for shared settings:

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
model = "gpt-4"
timeout = 30

[security]
require_api_key = true
allowed_keys = ["tenant-prod", "tenant-dev"]
workspace_roots = ["/repos/specs"]

[feature_flags]
enabled = ["environment_tools", "spec_helpers", "planning_tools", "response_contract_v2"]
```

## 🚀 Getting started

### Launch as an MCP server

```bash
foundry-mcp
```

The server will advertise its capabilities, feature flags, and response contract so MCP clients (Claude Code, Gemini CLI, etc.) can connect automatically.

### Use the native SDD CLI

```bash
python -m foundry_mcp.cli task next --specs-dir /path/to/specs
```

All CLI commands output JSON for reliable parsing by AI coding tools and mirror the legacy `claude-sdd-toolkit` surface.

### Claude Code setup

Add foundry-mcp through Claude Code settings (Command Palette → **Claude Code: Configure MCP Servers**) and include:

```json
{
  "mcpServers": {
    "foundry-mcp": {
      "command": "uvx",
      "args": ["foundry-mcp"],
      "env": {
        "FOUNDRY_MCP_SPECS_DIR": "/path/to/specs",
        "FOUNDRY_MCP_RESPONSE_CONTRACT": "v2"
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

### Quick usage examples

```bash
# List specs via MCP tool (unified router)
echo '{"action": "list"}' | foundry-mcp --tool spec

# Validate a spec via MCP tool
echo '{"action": "validate", "spec_id": "sdd-core-operations-2025-11-27-001"}' | foundry-mcp --tool spec

# Run CLI validation without an MCP client
python -m foundry_mcp.cli --specs-dir ./specs validate check sdd-core-operations-2025-11-27-001
```

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [SDD Philosophy](docs/concepts/sdd-philosophy.md) | Why spec-driven development matters |
| [Architecture Overview](docs/architecture/adr-001-cli-architecture.md) | CLI/MCP architecture decision record |
| [Development Guide](docs/guides/development-guide.md) | Setup, architecture, contributing |
| [Testing Guide](docs/guides/testing.md) | Running and debugging tests / fixtures |
| [LLM Configuration](docs/guides/llm-configuration.md) | Provider setup & fallbacks |
| [MCP Best Practices](docs/mcp_best_practices/README.md) | Canonical implementation checklist |
| [Response Schema](docs/codebase_standards/mcp_response_schema.md) | Standardized envelope reference |
| [CLI Output Contract](docs/codebase_standards/cli-output.md) | JSON-first CLI expectations |

## 🧪 Testing & quality gates

```bash
pytest                          # Full suite
pytest tests/integration/test_mcp_smoke.py  # MCP smoke tests
pytest tests/integration/test_mcp_tools.py  # Tool contract coverage
```

- Regression tests keep MCP/CLI adapters aligned with the legacy claude-sdd-toolkit contracts.
- Golden fixtures (`tests/fixtures/golden`) ensure response envelopes, error semantics, and pagination never regress.
- Freshness checks (doc generation, capability manifests) run alongside core unit and integration suites.

## 🤝 Contributing

Contributions are welcome! Please read the [MCP Best Practices](docs/mcp_best_practices/README.md) before submitting PRs. All changes should keep specs, docs, code, and fixtures in sync and follow the decision matrix in `AGENTS.md`.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built by [Tyler Burleigh](https://github.com/tylerburleigh)** · [Report an Issue](https://github.com/tylerburleigh/foundry-mcp/issues) · [View on GitHub](https://github.com/tylerburleigh/foundry-mcp)
