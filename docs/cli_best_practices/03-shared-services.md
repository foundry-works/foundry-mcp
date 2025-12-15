# 3. Shared Services Integration

> The CLI should call the same helpers as MCP tools so behavior never forks.

## 3.1 Responses & Serialization

- Create all payloads via `foundry_mcp.core.responses.success_response` / `error_response` and only convert to dict at the boundary.
- Adhere to the schema from [docs/codebase_standards/mcp_response_schema.md](../codebase_standards/mcp_response_schema.md); if a CLI command needs extra metadata, add it there first.

## 3.2 Validation & Security

- Reuse `foundry_mcp.core.security` input validation helpers. Do not parse/validate raw JSON in CLI code unless there is a shared helper.
- Apply the hygiene steps from [docs/mcp_best_practices/04-validation-input-hygiene.md](../mcp_best_practices/04-validation-input-hygiene.md) (sanitize paths, reject unexpected arguments, enforce allowlists for subprocesses).

## 3.3 Observability & Metrics

- All commands should register spans or events through `foundry_mcp.core.observability.audit_log` / `get_metrics` so dashboards capture CLI traffic alongside MCP usage.
- When commands interact with external tools (git, test runners), log command lines with sensitive data redacted per [docs/mcp_best_practices/08-security-trust-boundaries.md](../mcp_best_practices/08-security-trust-boundaries.md).

## 3.4 Resilience & Concurrency

- Use `foundry_mcp.core.resilience` timeouts and retry helpers; never implement ad-hoc sleeps or busy loops.
- Long-running operations (repo-local code scans, large spec/test runs) must opt into the same concurrency controls described in [docs/mcp_best_practices/15-concurrency-patterns.md](../mcp_best_practices/15-concurrency-patterns.md) so the CLI does not saturate resources when run alongside MCP workloads.

## 3.5 Discovery & Feature Flags

- Whenever a CLI feature ships, add corresponding entries to `foundry_mcp/core/discovery.py` and `mcp/capabilities_manifest.json` so clients can detect readiness.
- Document feature-flag lifecycles per [docs/mcp_best_practices/14-feature-flags.md](../mcp_best_practices/14-feature-flags.md); even if only the CLI consumes the flag today, the MCP adapters may depend on it tomorrow.
