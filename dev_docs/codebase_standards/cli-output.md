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

## 17.4 Token Management Warnings

Research workflows emit structured warnings when token budget management affects content. These warnings align with the `meta.warnings` and `meta.warning_details` schema defined in [mcp_response_schema.md](./mcp_response_schema.md).

### Warning Codes

| Code | Severity | Description |
|------|----------|-------------|
| `CONTENT_TRUNCATED` | info | Content was summarized or compressed to fit budget |
| `CONTENT_DROPPED` | warning | Low-priority content was removed entirely |
| `TOKEN_BUDGET_FLOORED` | info | Item preserved due to minimum items guardrail |
| `ARCHIVE_WRITE_FAILED` | warning | Content archive write failed (workflow continues) |
| `ARCHIVE_DISABLED` | info | Content archival disabled (missing directory or permissions) |
| `ARCHIVE_READ_CORRUPT` | warning | Archived content file was corrupted during read |
| `TOKEN_COUNT_ESTIMATE_USED` | info | Character-based heuristic used (tiktoken unavailable) |

### Example CLI Output (JSON)

```json
{
  "success": true,
  "data": {
    "task_id": "task-001",
    "report": "..."
  },
  "error": null,
  "meta": {
    "version": "response-v2",
    "content_fidelity": "partial",
    "content_fidelity_schema_version": "1.0",
    "dropped_content_ids": ["src-015", "src-022"],
    "warnings": [
      "CONTENT_TRUNCATED: Item src-003 truncated from 5000 to 2000 tokens",
      "CONTENT_DROPPED: Item src-015 dropped (priority: 0.2, budget exhausted)",
      "CONTENT_DROPPED: Item src-022 dropped (priority: 0.15, budget exhausted)"
    ],
    "warning_details": [
      {
        "code": "CONTENT_TRUNCATED",
        "severity": "info",
        "message": "Item src-003 truncated from 5000 to 2000 tokens",
        "context": {
          "item_id": "src-003",
          "original_tokens": 5000,
          "current_tokens": 2000,
          "fidelity_level": "condensed"
        }
      },
      {
        "code": "CONTENT_DROPPED",
        "severity": "warning",
        "message": "Item src-015 dropped (priority: 0.2, budget exhausted)",
        "context": {
          "item_id": "src-015",
          "priority": 0.2,
          "reason": "budget_exhausted"
        }
      }
    ]
  }
}
```

### Example CLI Output (Text Summary)

When `--json` is not supplied, token management warnings appear as a summary:

```
Research complete: research-001

Token Budget Summary:
  - Content fidelity: partial (65%)
  - 3 items truncated
  - 2 items dropped (archived)

Warnings:
  [info] CONTENT_TRUNCATED: 3 sources compressed to fit budget
  [warn] CONTENT_DROPPED: 2 low-priority sources removed

Report saved to: ./reports/research-001.md
```

### Content Fidelity in Reports

The `meta.content_fidelity` field indicates response completeness:

| Level | Description | Action |
|-------|-------------|--------|
| `full` | All content at original fidelity | None needed |
| `partial` | Some content summarized or dropped | Check `dropped_content_ids` |
| `summary` | Heavily condensed response | Consider re-running with larger model |
| `reference_only` | Only IDs, no content bodies | Retrieve content separately |

When `content_fidelity` is not `full`, check `meta.dropped_content_ids` for omitted items and `meta.content_archive_hashes` for retrieval hashes.

## 17.5 Testing Implications

- Golden tests should compare the JSON payloads, not the human-readable text. Text snapshots are optional and only for commands where Markdown output is part of the contract.
- When adding new CLI functionality, update the relevant spec node and reference this document to show compliance.
- Token management warning tests should verify both `meta.warnings` (string array) and `meta.warning_details` (structured) formats.
