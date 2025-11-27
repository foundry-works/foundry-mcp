# SDD Operations Not Yet Covered

The following `sdd` CLI commands still lack parity coverage or adapter support inside this repo. Grouping them by domain makes it easier to plan incremental additions.

---

## Descoped Operations (Claude Code-Specific)

These operations are **out of scope** for foundry-mcp because they assume the Claude Code plugin runtime environment:

| Operation | Reason |
|-----------|--------|
| `session-marker` | Manages cross-command session metadata - Claude Code conversation state |
| `get-work-mode` | Surfaces Claude Code's work mode toggle - plugin UI state |
| `context` | Reports CLI context - assumes Claude Code CLI environment |
| `skills-dev` | Meta-development for Claude Code plugins |
| `test` (skills/doc pipelines) | Test harnesses for Claude Code skill development |

**Note**: Some functionality can be adapted via MCP configuration (see "Feature Adaptations" section below).

## Recently Covered via MCP Best Practices Remediation

`specs/completed/mcp-best-practices-remediation-2025-11-26-001.json` shipped the missing MCP guardrails. The following SDD operations now have working adapters and should not be treated as backlog items:

- **Spec discovery and pagination** (`find-specs`, `list-specs`, `query-tasks`) via `src/foundry_mcp/tools/queries.py` with the new pagination helpers.
- **Test discovery** (`find-tests`) via `test_discover` in `src/foundry_mcp/tools/testing.py`.
- **Spec validation and stats** (`validate-spec`, `stats`) via `src/foundry_mcp/tools/validation.py` (`spec_validate`, `spec_stats`, `spec_validate_fix`).
- **Task prep & status reporting** (`prepare-task`, `status-report`) via `src/foundry_mcp/tools/tasks.py` (`task_prepare`, `task_progress`).
- **Journaling hygiene** (`check-journaling`) via `journal_list_unjournaled` in `src/foundry_mcp/tools/journal.py`.
- **Rendering/export** (`render`) via `src/foundry_mcp/tools/rendering.py` (`spec_render`, `spec_render_progress`).

Use these references when evaluating what actually remains to be wrapped.

---

## In-Scope Operations

### Environment & Project Setup
- `sdd-verify-toolchain` (was `verify-tools`) – sanity-check local CLI/toolchain availability.
- `sdd-init-workspace` (was `init-env`) – bootstrap the working directory for SDD workflows.
- `spec-detect-topology` (was `detect-project`) – auto-detect repository layout for specs/docs.
- `sdd-verify-environment` (was `check-environment`) – validate required OS packages, runtimes, and credentials.

### Spec Discovery & Validation Helpers
- `spec-find-related-files` (was `find-related-files`) – locate files referenced by a spec node.
- `spec-find-patterns` (was `find-pattern`) – search specs for structural/code patterns.
- `spec-detect-cycles` (was `find-circular-deps`) – detect cyclic task dependencies.
- `spec-validate-paths` (was `validate-paths`) – ensure file references exist on disk.

### Authoring, Editing & Metadata
- `spec-create` (was `create`) – scaffold a brand-new spec from scratch.
- `spec-template` (was `template`) – emit opinionated templates/snippets for spec sections.
- `spec-analyze` (was `analyze`) – perform deep heuristics on an existing spec.
- `spec-analyze-deps` (was `analyze-deps`) – inspect dependency graph health.
- `spec-apply-plan` (was `apply-modifications`) – apply bulk structural edits from a diff/plan.
- `review-parse-feedback` (was `parse-review`) – transform review feedback into structured actions.
- `spec-update-frontmatter` (was `update-frontmatter`) – mutate top-level metadata blocks.
- `task-add` / `task-remove` (were `add-task` / `remove-task`) – CRUD operations for tasks within specs.
- `assumption-add` / `assumption-list` (were `add-assumption` / `list-assumptions`) – manage assumption blocks.
- `revision-add` (was `add-revision`) – append revision history entries.
- `verification-add` / `verification-execute` / `verification-format-summary` (were `add-verification` / `execute-verify` / `format-verification-summary`) – manage verification artifacts attached to tasks/specs.
- `task-update-estimate` (was `update-estimate`) – tweak effort/time estimates per task.
- `spec-sync-metadata` (was `sync-metadata`) – push/pull spec metadata across stores.
- `task-update-metadata` (was `update-task-metadata`) – mutate per-task metadata payloads.
- `task-create-commit` (was `create-task-commit`) – generate Git commits for task-scoped changes.
- `journal-bulk-add` (was `bulk-journal`) – add multiple journal entries in one shot.

### Task Planning & Execution Utilities
- `plan-format` (was `format-plan`) – pretty-print task plans for sharing.
- `phase-list` (was `list-phases`) – enumerate phases in a spec.
- `phase-check-complete` (was `check-complete`) – verify completion readiness for a phase/spec.
- `phase-report-time` (was `phase-time`) – summarize timelines per phase.
- `spec-reconcile-state` (was `reconcile-state`) – compare file system vs. spec state for drift.
- `plan-report-time` (was `time-report`) – summarize time tracking metrics.
- `spec-audit` (was `audit-spec`) – run higher-level audits beyond basic validation.

### Lifecycle & Dependency Extras
Lifecycle mutation commands (assumption CRUD, revisions, verifications, estimate updates, task creation/deletion) remain unimplemented even after the remediation work. They rely on the same primitives listed in the Authoring section but need dedicated adapters because they affect dependency graphs and journaling requirements.

### Validation, Reporting & Analytics
- `spec-report` (was `report`) – produce human-readable validation/analysis reports (stat collection now covered via `spec_stats`, but higher-level reporting still missing).

### Collaboration, Review & PR Workflow (LLM-Powered)
These operations require LLM integration - users configure their provider via MCP config:

- `spec-review` (was `review`) – interactive or automated spec review sessions.
- `review-list-plan-tools` (was `list-plan-review-tools`) – enumerate review toolchains.
- `review-list-tools` (was `list-review-tools`) – list available review pipelines (non-plan-focused).
- `pr-create-with-spec` (was `create-pr`) – scaffold GitHub pull requests with SDD context (AI-enhanced descriptions).

### Documentation, Rendering & Skills (LLM-Powered)
- `spec-doc` (was `doc`) – generate human-facing documentation bundles.
- `spec-doc-llm` (was `llm-doc-gen`) – LLM-powered doc generation suite.
- `spec-review-fidelity` (was `fidelity-review`) – specialized documentation fidelity checks (code-to-spec comparison).

### Miscellaneous Tooling
- `sdd-cache-manage` (was `cache`) – manage SDD CLI cache entries.
- `spec-schema-export` (was `schema`) – emit JSON schemas for specs/tasks.

---

## Feature Adaptations for MCP

Some Claude Code features can be adapted to work in the MCP framework via configuration:

| Claude Code Feature | MCP Adaptation |
|---------------------|----------------|
| Work mode toggle | `FOUNDRY_MCP_WORK_MODE` env var or `[workflow].mode` in TOML |
| Session context | `get_server_context` MCP tool (exposes workspace info) |
| LLM provider selection | `FOUNDRY_MCP_LLM_*` env vars or `[llm]` section in TOML |

### Proposed Config Extensions

```toml
# foundry-mcp.toml additions

[llm]
provider = "openai"  # or "anthropic", "local", etc.
api_key = ""         # or use FOUNDRY_MCP_LLM_API_KEY env var
model = "gpt-4"      # default model
timeout = 120

[workflow]
mode = "standard"    # or "strict", "interactive", etc.
auto_validate = true
journal_enabled = true
```

### Environment Variables

```bash
# LLM Configuration
FOUNDRY_MCP_LLM_PROVIDER=openai
FOUNDRY_MCP_LLM_API_KEY=sk-...
FOUNDRY_MCP_LLM_MODEL=gpt-4
FOUNDRY_MCP_LLM_TIMEOUT=120

# Workflow Mode
FOUNDRY_MCP_WORK_MODE=standard
FOUNDRY_MCP_AUTO_VALIDATE=true
```

---

## Implementation Notes

### Response Schema Standardization Guardrails
Reference: `specs/completed/response-schema-standardization-2025-11-26-001.json` (authoritative contract) and `src/foundry_mcp/core/responses.py` (helper implementation). As of that rollout, all shipped tool modules—`tasks.py`, `queries.py`, `docs.py`, `testing.py`, `journal.py`, `validation.py`, `server.py`, `lifecycle.py`, and `rendering.py`—already import and return `success_response` / `error_response`. Keep the backlog honest by applying the same pattern to every new adapter listed above.

To avoid another retrofit cycle, every new MCP operation or adapter should:

1. **Use the shared helpers** once `foundry_mcp.core.responses` lands (`success_response` / `error_response`). No tool should shape ad-hoc dicts.
2. **Emit the v2 envelope** (`success`, `data`, `error`, `meta.version=response-v2`) even when a tool currently returns no payload—set `data={}` and `error=null`.
3. **Document `meta` usage** when adding pagination, warnings, or telemetry flags so downstream clients can rely on consistent naming.
4. **Add regression tests** (or extend `tests/integration/test_mcp_tools.py`) that assert the response envelope before marking an operation “covered.”
5. **Note deviations in specs**: if an operation legitimately streams multiple payloads, capture the exception rationale in its spec so future refactors don’t “fix” it.

Helper-level enforcement lives in `tests/test_responses.py`; adapter-level coverage should extend the relevant suites (typically `tests/integration/test_mcp_tools.py`) whenever you claim a new operation from this backlog.

### MCP Best Practices Remediation Guardrails
Reference: `specs/completed/mcp-best-practices-remediation-2025-11-26-001.json`. Any new adapter work must reuse the infrastructure that spec delivered:

1. **Input hygiene & prompt shielding** – use `foundry_mcp/core/security.py` (`MAX_INPUT_SIZE`, `validate_input_size`, `detect_prompt_injection`) and redact via `foundry_mcp/core/observability.py` before emitting logs.
2. **Feature gating & telemetry** – register new operations with `foundry_mcp/core/feature_flags.py` and expose readiness through the discovery endpoints so MCP clients can negotiate safely.
3. **Resilience & concurrency** – wrap outbound or long-running work with `foundry_mcp/core/resilience.py` (`with_timeout`, `retry_with_backoff`, `CircuitBreaker`) and `foundry_mcp/core/concurrency.py` to enforce rate limits/cancellation.
4. **Pagination & discovery helpers** – all list-style responses must use `foundry_mcp/core/pagination.py` + `paginated_response`, and expose metadata through `foundry_mcp/core/discovery.py`/`src/foundry_mcp/tools/discovery.py`.
5. **LLM-friendly output** – honor `foundry_mcp/core/llm_patterns.py` (`progressive_disclosure`, `batch_operation`) when shaping verbose responses so they degrade gracefully for LLM consumers.

### LLM-Powered Features
For operations requiring AI (review, llm-doc-gen, create-pr, fidelity-review, render with insights):

1. **Provider Configuration**: Users configure their preferred LLM provider via MCP server config
2. **Provider Abstraction**: Use an abstraction layer supporting multiple providers (OpenAI, Anthropic, local models)
3. **Graceful Degradation**: Features should work in "data-only" mode if no LLM is configured

### Server Context Tool
Instead of Claude Code's `context` command, expose an MCP tool:

```python
@mcp.tool()
def get_server_context() -> dict:
    """Get current server configuration and workspace context."""
    return {
        "workspace_roots": config.workspace_roots,
        "specs_dir": str(config.specs_dir),
        "work_mode": config.work_mode,
        "llm_provider": config.llm_provider,
        "llm_configured": bool(config.llm_api_key),
    }
```

---

> Use this file as the authoritative backlog when deciding which SDD commands to wrap next. For each entry, add parity fixtures/tests plus adapter plumbing before marking it "covered."
