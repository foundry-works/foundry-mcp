# 16. Tool Naming Conventions

> Provide predictable, LLM-friendly tool names that encode scope and verb semantics.

## Why Naming Matters

Consistent naming shortens discovery time, improves LLM selection accuracy, and keeps parity between the SDD CLI and MCP adapters. This guidance extends [Tool Metadata & Discovery](../mcp_best_practices/13-tool-discovery.md) with concrete conventions for SDD-oriented operations.

## Principles

1. **Verb–Noun Order** – Start with the action (`review-parse`, `task-update`), followed by the subject. This groups related verbs alphabetically in discovery listings.
2. **Domain Prefixes** – Use short prefixes to signal the artifact being touched:
   - `spec-` for spec-wide operations (authoring, validation, reporting)
   - `task-` for task-scoped mutation or inspection
   - `plan-` / `phase-` for planning utilities
   - `review-` for review workflows
   - `verification-`, `assumption-`, `revision-`, `journal-` for lifecycle domains
   - `sdd-` for environment-wide helpers (bootstrap, cache, config)
3. **Hyphen Separators** – Prefer `kebab-case` so MCP tool registries remain readable and avoid camelCase drift across adapters.
4. **Status-Quo Mapping** – When migrating existing tools, surface the new name while aliasing the legacy command until clients update, then deprecate per [§13 Tool Discovery](../mcp_best_practices/13-tool-discovery.md#deprecation-handling).
5. **No Overloaded Roots** – Avoid reusing generic verbs (`create`, `init`) without prefixes; collisions make tool chaining brittle for LLMs.

## Recommended Mapping Matrix

| Domain | Prefix | Examples |
|--------|--------|----------|
| Environment & bootstrap | `sdd-` | `sdd-verify-toolchain`, `sdd-init-workspace`, `sdd-cache-manage` |
| Spec-wide authoring | `spec-` | `spec-create`, `spec-update-frontmatter`, `spec-reconcile-state` |
| Task lifecycle | `task-` | `task-add`, `task-update-metadata`, `task-create-commit` |
| Planning & phasing | `plan-` / `phase-` | `plan-format`, `plan-report-time`, `phase-check-complete` |
| Review / PR flows | `review-` / `pr-` | `review-list-tools`, `review-parse-feedback`, `pr-create-with-spec` |
| Lifecycle extras | domain noun | `assumption-list`, `verification-add`, `journal-bulk-add` |

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

## Current Implementation Audit *(2025-11-27)*

| Module | Canonical Tools | Status | Notes |
|--------|-----------------|--------|-------|
| `server.py` | `sdd-server-capabilities`, `spec-list-basic`, `spec-get`, `spec-get-hierarchy`, `task-get` | ✅ live | Registered via `canonical_tool` with canonical names only.
| `tools/queries.py` | `spec-find`, `spec-list`, `task-query` | ✅ live | Canonical names registered across CLI + discovery listings.
| `validation.py` | `spec-validate`, `spec-fix`, `spec-stats`, `spec-validate-fix` | ✅ live | Canonical names ship via `canonical_tool`; no legacy aliases remain.
| `rendering.py` | `spec-render`, `spec-render-progress`, `task-list` | ✅ live | Rendering helpers expose canonical names for registry + docs.
| `journal.py` | `journal-add`, `journal-list`, `task-block`, `task-unblock`, `task-list-blocked`, `journal-list-unjournaled` | ✅ live | Lifecycle + journaling helpers now use canonical prefixes only.
| `docs.py` | `code-find-class`, `code-find-function`, `code-trace-calls`, `code-impact-analysis`, `code-get-callers`, `code-get-callees`, `doc-stats` | ✅ live | Code-analysis suite follows `code-`/`doc-` naming for easier discovery.
| `discovery.py` | `tool-list`, `tool-get-schema`, `capability-get`, `capability-negotiate`, `tool-list-categories` | ✅ live | Discovery endpoints expose canonical tool/capability prefixes.
| `lifecycle.py` | `spec-lifecycle-move`, `spec-lifecycle-activate`, `spec-lifecycle-complete`, `spec-lifecycle-archive`, `spec-lifecycle-state`, `spec-list-by-folder` | ✅ live | All lifecycle transitions aligned to `spec-lifecycle-*` verbs.
| `tasks.py` | `task-prepare`, `task-next`, `task-info`, `task-check-deps`, `task-update-status`, `task-complete`, `task-start`, `task-progress` | ✅ live | Task operations standardized under canonical prefixes.
| `testing.py` | `test-run`, `test-discover`, `test-presets`, `test-run-quick`, `test-run-unit` | ✅ live | Testing adapters advertise canonical names for preset + runner variants.

## Related Documents

- [Tool Metadata & Discovery](../mcp_best_practices/13-tool-discovery.md)
- [Versioned Contracts](../mcp_best_practices/01-versioned-contracts.md)
- [Spec-Driven Development](../mcp_best_practices/09-spec-driven-development.md)

---

**Navigation:** [← Feature Flags & Rollouts](../mcp_best_practices/14-feature-flags.md) | [Index](../mcp_best_practices/README.md)
