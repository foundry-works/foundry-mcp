# Configuration

foundry-mcp supports configuration via TOML and environment variables. The CLI
and MCP server share the same settings.

## Configuration order

Configuration is loaded in layers, with each layer overriding the previous:

1. **Defaults** - Built-in default values
2. **User config** - `~/.foundry-mcp.toml` (optional, user-wide settings)
3. **Project config** - `./foundry-mcp.toml` (optional, project-specific)
4. **Environment variables** - Runtime overrides (highest priority)

### Config file locations

| Location | Purpose | Example use cases |
|----------|---------|-------------------|
| `~/.foundry-mcp.toml` | User defaults | API keys, preferred LLM providers, logging preferences |
| `./foundry-mcp.toml` | Project settings | specs_dir, workspace roots, project-specific tool config |

### Legacy compatibility

For backwards compatibility, if `./foundry-mcp.toml` doesn't exist, the system
will fall back to `./.foundry-mcp.toml` (dot-prefixed) in the project directory.

## Minimal TOML example

```toml
[workspace]
specs_dir = "./specs"

[logging]
level = "INFO"

[llm]
provider = "openai"
model = "gpt-4.1"
timeout = 30
```

## Common environment variables

| Variable | Purpose |
| --- | --- |
| `FOUNDRY_MCP_SPECS_DIR` | Override specs directory |
| `FOUNDRY_MCP_WORKSPACE_ROOTS` | Restrict allowed workspace roots |
| `FOUNDRY_MCP_LOG_LEVEL` | Set log level (INFO, DEBUG, etc.) |
| `FOUNDRY_MCP_API_KEYS` | Require API keys for tool access |
| `FOUNDRY_MCP_REQUIRE_AUTH` | Enforce auth on all tools |
| `FOUNDRY_MCP_FEATURE_FLAGS` | Enable feature flags |

## LLM providers

Quick setup for common providers:

- OpenAI: set `FOUNDRY_MCP_LLM_PROVIDER=openai` and `OPENAI_API_KEY`
- Anthropic: set `FOUNDRY_MCP_LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`
- Local: set `FOUNDRY_MCP_LLM_PROVIDER=local` and `FOUNDRY_MCP_LLM_BASE_URL`

Common LLM environment variables:

| Variable | Purpose |
| --- | --- |
| `FOUNDRY_MCP_LLM_PROVIDER` | Provider name (`openai`, `anthropic`, `local`) |
| `FOUNDRY_MCP_LLM_API_KEY` | Unified API key override |
| `FOUNDRY_MCP_LLM_MODEL` | Model identifier |
| `FOUNDRY_MCP_LLM_BASE_URL` | Custom API endpoint |
| `FOUNDRY_MCP_LLM_TIMEOUT` | Request timeout (seconds) |

LLM configuration is documented in detail here:

- [LLM Configuration Guide](guides/llm-configuration.md)

## Specs directory resolution

If you do not set `FOUNDRY_MCP_SPECS_DIR`, the CLI and server will attempt to
auto-detect a `specs/` directory in the workspace.
