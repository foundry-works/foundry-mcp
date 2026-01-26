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

## Research Configuration

The `[research]` section controls deep research workflows including search provider
settings. For full configuration options, see `samples/foundry-mcp.toml`.

### Tavily Search Provider

Tavily is a web search provider optimized for AI applications. Configure via
environment variable or TOML:

```bash
export TAVILY_API_KEY="tvly-..."
```

#### Search Parameters

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `tavily_search_depth` | string | `"basic"` | Search mode: `"basic"`, `"advanced"` (2x credits), `"fast"`, `"ultra_fast"` |
| `tavily_topic` | string | `"general"` | Search topic: `"general"`, `"news"` |
| `tavily_news_days` | int | `null` | Days limit for news (1-365, only when `topic="news"`) |
| `tavily_include_images` | bool | `false` | Include image results |
| `tavily_country` | string | `null` | ISO 3166-1 alpha-2 code to boost results (e.g., `"US"`) |
| `tavily_chunks_per_source` | int | `3` | Chunks per source for advanced search (1-5) |
| `tavily_auto_parameters` | bool | `false` | Let Tavily auto-configure based on query |

#### Extract Parameters

Tavily Extract enables URL content extraction for deeper analysis.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `tavily_extract_depth` | string | `"basic"` | Extract mode: `"basic"`, `"advanced"` |
| `tavily_extract_include_images` | bool | `false` | Include images in extraction |
| `tavily_extract_in_deep_research` | bool | `false` | Enable extract as follow-up step |
| `tavily_extract_max_urls` | int | `5` | Max URLs to extract per deep research run |

#### Research Mode Smart Defaults

When using deep research, parameters are adjusted based on `deep_research_mode`:

| Mode | Search Depth | Source Prioritization |
|------|-------------|----------------------|
| `"general"` | `basic` | No preference |
| `"academic"` | `advanced` | Journals, publishers, preprints |
| `"technical"` | `advanced` | Official docs, arxiv, Stack Overflow |

#### Example Configuration

```toml
[research]
# Search provider credentials (prefer env vars in production)
# tavily_api_key = "tvly-..."

# Search parameters
tavily_search_depth = "basic"      # "basic", "advanced" (2x credits), "fast", "ultra_fast"
tavily_topic = "general"           # "general", "news"
tavily_news_days = 7               # only when topic = "news"
tavily_include_images = false
tavily_country = "US"              # boost results from country
tavily_chunks_per_source = 3       # 1-5, for advanced search
tavily_auto_parameters = false     # let Tavily auto-configure

# Extract parameters
tavily_extract_depth = "basic"           # "basic", "advanced"
tavily_extract_include_images = false
tavily_extract_in_deep_research = false  # enable extract follow-up
tavily_extract_max_urls = 5              # max URLs per deep research run

# Deep research mode affects Tavily parameter selection
deep_research_mode = "technical"   # "general", "academic", "technical"
```

#### Credit Cost Awareness

- `search_depth="basic"` - Standard credit cost
- `search_depth="advanced"` - 2x credit cost (use for deeper analysis)
- `search_depth="fast"` / `"ultra_fast"` - Reduced latency, standard cost

## Specs directory resolution

If you do not set `FOUNDRY_MCP_SPECS_DIR`, the CLI and server will attempt to
auto-detect a `specs/` directory in the workspace.
