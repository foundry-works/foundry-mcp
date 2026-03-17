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
| `~/.foundry-mcp.toml` | User defaults | Preferred CLI providers, logging preferences |
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

[consultation]
priority = ["[cli]claude:opus", "[cli]gemini:pro"]
```

## Common environment variables

| Variable | Purpose |
| --- | --- |
| `FOUNDRY_MCP_SPECS_DIR` | Override specs directory |
| `FOUNDRY_MCP_WORKSPACE_ROOTS` | Restrict allowed workspace roots |
| `FOUNDRY_MCP_LOG_LEVEL` | Set log level (INFO, DEBUG, etc.) |
| `FOUNDRY_MCP_API_KEYS` | Require API keys for tool access |
| `FOUNDRY_MCP_REQUIRE_AUTH` | Enforce auth on all tools |
| `FOUNDRY_MCP_FEATURE_FLAGS` | Bulk feature flag overrides (`flag` or `flag=true/false`, comma-separated) |
| `FOUNDRY_MCP_FEATURE_FLAG_<NAME>` | Per-flag override (highest precedence), e.g. `FOUNDRY_MCP_FEATURE_FLAG_AUTONOMY_SESSIONS=true` |
| `FOUNDRY_MCP_AUTONOMY_POSTURE` | Posture profile (`unattended`, `supervised`, `debug`) |
| `FOUNDRY_MCP_AUTONOMY_DEFAULT_GATE_POLICY` | Default `session-start` gate policy (`strict`, `lenient`, `manual`) |
| `FOUNDRY_MCP_AUTONOMY_DEFAULT_STOP_ON_PHASE_COMPLETION` | Default `stop_on_phase_completion` for session starts |
| `FOUNDRY_MCP_AUTONOMY_DEFAULT_AUTO_RETRY_FIDELITY_GATE` | Default `auto_retry_fidelity_gate` for session starts |

## Feature Flags

Feature flags can be set in TOML and overridden via environment variables.
Precedence (highest to lowest):

1. `FOUNDRY_MCP_FEATURE_FLAG_<NAME>` per-flag env vars
2. `FOUNDRY_MCP_FEATURE_FLAGS` bulk env var
3. `[feature_flags]` in TOML

Example TOML:

```toml
[feature_flags]
autonomy_sessions = true
autonomy_fidelity_gates = false
```

Example env overrides:

```bash
export FOUNDRY_MCP_FEATURE_FLAGS="autonomy_sessions=true,autonomy_fidelity_gates=true"
export FOUNDRY_MCP_FEATURE_FLAG_AUTONOMY_FIDELITY_GATES=false
```

Startup validation emits warnings for inconsistent combinations (for example,
`autonomy_fidelity_gates=true` while `autonomy_sessions=false`).

## Autonomy Posture Profiles

Posture profiles provide one-knob defaults for autonomy security and session behavior.
Profiles are a fixed enum:

- `unattended`: headless-safe defaults (`autonomy_runner`, escape hatches closed)
- `supervised`: human-in-the-loop defaults (guardrails on, escape hatches available with reason codes)
- `debug`: maximum flexibility for manual troubleshooting (no unattended loops)

Example TOML:

```toml
[autonomy_posture]
profile = "unattended"
```

Profile defaults can be overridden through direct configuration:

```toml
[autonomy_security]
role = "maintainer"
allow_lock_bypass = true
allow_gate_waiver = true

[autonomy_session_defaults]
gate_policy = "manual"
stop_on_phase_completion = false
```

When a profile and direct overrides conflict, startup warnings call out unsafe combinations (for example, unattended posture with maintainer role or bypass enabled).

## LLM providers

foundry-mcp uses CLI-based providers (claude, gemini, codex, cursor-agent, opencode) configured via the `[consultation]` section. See the [LLM Configuration Guide](guides/llm-configuration.md) for full details.

Common consultation environment variables:

| Variable | Purpose |
| --- | --- |
| `FOUNDRY_MCP_CONSULTATION_PRIORITY` | Comma-separated provider priority list |
| `FOUNDRY_MCP_CONSULTATION_TIMEOUT` | Default timeout in seconds |
| `FOUNDRY_MCP_CONSULTATION_MAX_RETRIES` | Max retry attempts |
| `FOUNDRY_MCP_CONSULTATION_FALLBACK_ENABLED` | Enable provider fallback |

## Secret Management

The autonomy subsystem uses a server secret for HMAC-based integrity protection
of gate evidence. The secret ensures that gate verdicts cannot be tampered with
between the review step and the evidence-submission step.

### How the secret works

- **Auto-generated** on first use: a 32-byte random key is created automatically
- **Stored** at `$FOUNDRY_DATA_DIR/.server_secret` (default: `~/.foundry-mcp/.server_secret`) with mode `0600` (owner read/write only)
- **Cached** in memory after first load for performance
- **Override** via `FOUNDRY_MCP_GATE_SECRET` environment variable (for deterministic testing)

### Secret rotation / recovery procedure

If the server secret is compromised or needs rotation:

1. **Stop the server** — ensure no autonomous sessions are actively running
2. **Delete the secret file**:
   ```bash
   rm ~/.foundry-mcp/.server_secret
   # or if using custom data dir:
   rm $FOUNDRY_DATA_DIR/.server_secret
   ```
3. **Restart the server** — a new secret is auto-generated on first use

### Impact of rotation

- All **in-flight gate evidence** becomes invalid because HMAC checksums were
  computed with the old secret. The orchestrator will reject stale evidence and
  re-request gate steps.
- Active sessions with **pending manual gate acknowledgments** will need the
  gate step re-issued.
- Sessions should be **rebased** (`session-rebase`) after rotation to
  re-establish a clean state, or ended and restarted.
- Completed sessions and historical audit data are **not affected** — they are
  already persisted and do not require the secret for reads.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `FOUNDRY_DATA_DIR` | Override the data directory (default: `~/.foundry-mcp`) |
| `FOUNDRY_MCP_GATE_SECRET` | Provide a deterministic secret (for testing only — do not use in production) |

### Security notes

- Never commit the secret file to version control
- The secret file permissions are set to `0600` (owner-only) automatically
- In production, use filesystem-level access controls to protect `$FOUNDRY_DATA_DIR`
- Gate evidence is short-lived by design, limiting the window of exposure if
  a secret is compromised

## Specs directory resolution

If you do not set `FOUNDRY_MCP_SPECS_DIR`, the CLI and server will attempt to
auto-detect a `specs/` directory in the workspace.
