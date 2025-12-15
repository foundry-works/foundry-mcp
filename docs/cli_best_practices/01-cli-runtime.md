# 1. CLI Runtime Ergonomics

> Align CLI UX with MCP envelopes so LLM-initiated and human-initiated workflows behave the same.

## 1.1 Parsers & Entry Points

- Use the shared Click entry point in `foundry_mcp.cli.main` for every binary (`sdd`, `sdd-native`).
- Support the global options set in [docs/mcp_best_practices/13-tool-discovery.md](../mcp_best_practices/13-tool-discovery.md) (including `--json` and workspace overrides) so tooling can share scripts.
- Register command groups through the CLI registry (`foundry_mcp.cli.registry`) to keep help output consistent.

## 1.2 Verbosity & Output Filtering

- Map `--quiet`, default, and `--verbose` to the same enum used by MCP envelopes.
- Prefer JSON output as the canonical interface. Only emit plain text or lightweight Markdown summaries when explicitly requested (e.g., `--human` flag) and keep them free of color/ANSI formatting.
- Route any human-readable text through the shared printer abstraction so observability ([docs/mcp_best_practices/05-observability-telemetry.md](../mcp_best_practices/05-observability-telemetry.md)) can tap into the same structured logs; do **not** introduce CLI-only rich UI layers.
- When `--json` is set, call `foundry_mcp.core.responses.success_response` / `error_response` and run the payload through the `prepare_output` filter defined for the equivalent MCP tool. This keeps field inclusion rules synchronized with [docs/codebase_standards/mcp_response_schema.md](../codebase_standards/mcp_response_schema.md).

## 1.3 Workspace Detection & Paths

- Resolve specs, reports, and cache directories using the shared helpers under `foundry_mcp.core.paths` (avoid `os.getcwd()` directly).
- CLI commands MUST surface absolute paths in JSON mode and friendly relative hints in text mode to match the environment tools.

## 1.4 Timeouts, Cancellation, and Resilience

- Wrap external subprocesses or long-running scans with the decorators from `foundry_mcp.core.resilience` so CLI invocations inherit the same timeout budgets defined in [docs/mcp_best_practices/12-timeout-resilience.md](../mcp_best_practices/12-timeout-resilience.md).
- Propagate `KeyboardInterrupt` and cancellation tokens all the way to MCP helpers; do not swallow exceptions silently.

## 1.5 Metrics & Logging

- Emit structured logs via `foundry_mcp.core.observability` with context IDs pulled from CLI flags (or generated UUIDs). This ensures CLI-only usage still populates dashboards shared with MCP request traffic per [docs/mcp_best_practices/05-observability-telemetry.md](../mcp_best_practices/05-observability-telemetry.md).
- For long-running commands, add progress hooks that can be reused by the MCP streaming APIs (e.g., environment verification, repo-local code scans via `code(action=...)`).
