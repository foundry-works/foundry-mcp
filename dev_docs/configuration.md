# Configuration Reference

> Complete reference for foundry-mcp configuration options.

## Overview

foundry-mcp can be configured through:
1. **TOML file**: `foundry-mcp.toml` in workspace root
2. **Environment variables**: Prefixed with `FOUNDRY_MCP_`
3. **Default values**: Built-in defaults for all options

Priority: Environment variables > TOML file > Defaults

## Configuration File

Create `foundry-mcp.toml` in your workspace root:

```toml
[workspace]
specs_dir = "./specs"

[logging]
level = "INFO"
structured = true
```

## Environment Variables

All settings can be set via environment variables with `FOUNDRY_MCP_` prefix and uppercase.

## Validation

Configuration is validated at startup. The server will fail to start with invalid configuration.
