# 16. Tool Naming Conventions

> Provide predictable, LLM-friendly tool names that encode scope and verb semantics.

## Why Naming Matters

Consistent naming shortens discovery time, improves LLM selection accuracy, and keeps parity between the SDD CLI and MCP adapters. This guidance extends [Tool Metadata & Discovery](../mcp_best_practices/13-tool-discovery.md) with concrete conventions for SDD-oriented operations.

## Principles

1. **Router + Action** – Use a small set of domain routers (`spec`, `task`, `review`, etc.) and encode the verb in an `action` parameter. This keeps the advertised manifest small and LLM-friendly.
2. **Action Naming** – `action` values MUST be stable, `kebab-case`, and scoped to the router (e.g., `spec: validate`, `task: next`, `review: fidelity`).
3. **Avoid Verb Explosion** – Prefer extending an existing router with a new `action` instead of introducing a new top-level tool name.
4. **Deprecation Discipline** – If you temporarily support legacy tool names, document the retirement timeline in specs and remove them within two releases per [§13 Tool Discovery](../mcp_best_practices/13-tool-discovery.md#deprecation-handling).
5. **No Overloaded Actions** – Don’t reuse the same `action` string with incompatible semantics; add a new action or introduce a separate parameter for mode selection.

## Recommended Mapping Matrix

| Domain router | `action` examples | Notes |
|--------------|------------------|-------|
| `environment` | `init`, `verify-env`, `verify-toolchain`, `setup`, `detect` | Workspace + toolchain hygiene |
| `spec` | `find`, `list`, `validate`, `fix`, `stats`, `validate-fix`, `analyze`, `analyze-deps` | Discovery/validation/analysis |
| `authoring` | `spec-create`, `spec-template`, `spec-update-frontmatter`, `phase-add`, `phase-add-bulk`, `phase-remove`, `phase-move`, `phase-template`, `phase-update-metadata`, `assumption-add`, `assumption-list`, `revision-add`, `spec-find-replace`, `spec-rollback` | Spec mutations |
| `task` | `next`, `prepare`, `start`, `complete`, `progress`, `query`, `hierarchy`, `block`, `unblock` | Task execution surface |
| `lifecycle` | `activate`, `complete`, `archive`, `move`, `state` | Spec folder/state transitions |
| `journal` | `add`, `list`, `list-unjournaled` | Journal records |
| `test` | `run` (`preset=quick|unit|full`), `discover` | Pytest integration |
| `review` | `spec`, `fidelity`, `parse-feedback`, `list-tools`, `list-plan-tools` | LLM review workflows |
| `code` | `find-class`, `find-function`, `callers`, `callees`, `trace`, `impact` | Repo-local code navigation |
| `server` | `tools`, `schema`, `capabilities`, `context`, `llm-status` | Discovery/capabilities introspection |
| `health` | `liveness`, `readiness`, `check` | Operational health checks |

## CLI Naming Plan

### Binary & Entry Points

1. **`sdd`** – Primary end-user CLI that mirrors the canonical command set. Every doc, spec, and test assumes this name.
2. **`sdd-native`** – Development/testing alias wired to the same entry point. Useful for development environments or when multiple CLI versions need to coexist.
3. **`sdd-dev-*` helpers** – Temporary binaries (e.g., `sdd-dev-smoke`) may exist inside CI scaffolding but MUST remain unadvertised outside internal scripts.

All binaries import `foundry_mcp.sdd_cli.__main__` so the same parser/runtime stack powers both aliases.

### Subcommand Namespaces

| CLI Namespace | Scope | Unified router alignment |
|---------------|-------|--------------------------|
| `plan` | Spec creation + workspace analysis | `plan`, `authoring`, `environment` |
| `next` | Task discovery + preparation | `task` |
| `update` | Status changes, journaling, lifecycle | `task`, `journal`, `lifecycle` |
| `validate` | Validation, fix, reporting | `spec` |
| `provider` | LLM provider management | `provider` |
| `test` | Test discovery/execution | `test` |
| `spec-mod` | Bulk spec modifications | `authoring`, `verification` |
| `plan-review` / `pr` | Reviews and PR helpers | `review`, `pr` |
| `context` | Session + token tracking | `server` (context), `environment` |

### Operation Naming Workflow

1. Pick the CLI namespace first (table above) so `--help` output groups related verbs.
2. Choose the canonical tool prefix that best matches the artifact (from the Recommended Mapping Matrix).
3. Author the canonical MCP tool name (`prefix-verb`) and reuse that name in the CLI output/help text so LLMs see the same identifier everywhere.
4. Update specs/tests/docs simultaneously; do **not** ship CLI commands that differ from the canonical tool name unless a `--alias` flag is explicitly documented.

## Migration Checklist

1. **Choose Prefix** – Identify the narrowest artifact the tool acts upon and apply the matching prefix.
2. **Normalize Verb** – Use an imperative verb (`create`, `update`, `list`, `report`, `execute`). Prefer `report`/`format` over ambiguous `process`/`handle`.
3. **Canonical Only** – Register the canonical MCP name and update specs/tests simultaneously; do not ship parallel legacy identifiers.
4. **Document Updates** – Reflect the rename in:
   - Specs (`specs/completed/*.json`)
   - `docs/` references (including [OPERATIONS_TO_ADD.md](../../half_baked_plans/OPERATIONS_TO_ADD.md))
   - Tests and fixtures (see [§10 Testing & Fixtures](../mcp_best_practices/10-testing-fixtures.md))
5. **Communicate in Discovery** – Update tool descriptions, tags, and capability listings (`get_capabilities`, `list_tools`) so clients see the new canonical name.

## Existing Operations To Audit

- Confirm that all adapters under `src/foundry_mcp/tools/` use the prefixes above.
- Pay special attention to pre-guidance helpers that may still carry historical prefixes and ensure docs/specs/tests move together when renaming.

## Current Implementation Audit *(unified routers)*

The canonical advertised tool surface is the 17 unified routers in `mcp/capabilities_manifest.json`:

`health`, `plan`, `pr`, `error`, `metrics`, `journal`, `authoring`, `provider`, `environment`, `lifecycle`, `verification`, `task`, `spec`, `review`, `code`, `server`, `test`.

Each router exposes a stable `action` enum; add functionality by extending `action` (and updating the manifest/specs/tests) rather than introducing new top-level tool names.

## Related Documents

- [Tool Metadata & Discovery](../mcp_best_practices/13-tool-discovery.md)
- [Versioned Contracts](../mcp_best_practices/01-versioned-contracts.md)
- [Spec-Driven Development](../mcp_best_practices/09-spec-driven-development.md)

---

**Navigation:** [← Feature Flags & Rollouts](../mcp_best_practices/14-feature-flags.md) | [Index](../mcp_best_practices/README.md)
