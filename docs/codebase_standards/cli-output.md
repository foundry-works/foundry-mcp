# 17. CLI Output Standards

> Keep the CLI and MCP responses aligned by emitting the same envelopes and avoiding bespoke UI layers.

## 17.1 Canonical Output Modes

1. **JSON First** – Every command MUST support `--json` and return data using `foundry_mcp.core.responses.success_response` / `error_response`, conforming to the schema in [mcp_response_schema.md](./mcp_response_schema.md). The current `sdd` CLI always emits JSON envelopes; `--json` exists as an explicit compatibility flag.
2. **Text as a Convenience** – Plain-text or Markdown summaries MAY be emitted when `--json` is not supplied, but they must remain ANSI-free, deterministic, and derived from the same payload as the JSON response.
3. **No Rich UI** – Do not import or recreate pretty-print/rich-UI stacks (progress bars, colorized tables, etc.). Reuse the thin printer wrapper defined in [cli_best_practices/01-cli-runtime.md](../cli_best_practices/01-cli-runtime.md) so logging hooks stay consistent.

## 17.2 Shared Services

- Serialize everything via the shared helpers in `foundry_mcp.core.responses` and validation layers described in [../mcp_best_practices/02-envelopes-metadata.md](../mcp_best_practices/02-envelopes-metadata.md) and [../mcp_best_practices/03-serialization-helpers.md](../mcp_best_practices/03-serialization-helpers.md).
- Propagate observability metadata exactly as outlined in [../mcp_best_practices/05-observability-telemetry.md](../mcp_best_practices/05-observability-telemetry.md); CLI-only fields should never diverge from MCP logging schemas.

## 17.3 Flag & Help Expectations

- Document JSON-first behavior in each command's `--help` output (e.g., “Use `--json` for machine-readable envelopes; plain text is a thin summary only”).
- If a command provides Markdown output, gate it behind an explicit flag (`--markdown`) and describe its intended consumer (human reviewers, release notes, etc.).

## 17.4 Testing Implications

- Golden tests should compare the JSON payloads, not the human-readable text. Text snapshots are optional and only for commands where Markdown output is part of the contract.
- When adding new CLI functionality, update the relevant spec node and reference this document to show compliance.
