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

---

## In-Scope Operations

### Environment & Project Setup
- `verify-tools` – sanity-check local CLI/toolchain availability.
- `init-env` – bootstrap the working directory for SDD workflows.
- `detect-project` – auto-detect repository layout for specs/docs.
- `check-environment` – validate required OS packages, runtimes, and credentials.

### Spec Discovery & Validation Helpers
- `find-specs` – glob discovery of spec files by pattern.
- `find-related-files` – locate files referenced by a spec node.
- `find-pattern` – search specs for structural/code patterns.
- `find-tests` – map specs to likely test files.
- `find-circular-deps` – detect cyclic task dependencies.
- `validate-spec` – legacy validator variant with schema-centric output.
- `validate-paths` – ensure file references exist on disk.

### Authoring, Editing & Metadata
- `create` – scaffold a brand-new spec from scratch.
- `template` – emit opinionated templates/snippets for spec sections.
- `analyze` – perform deep heuristics on an existing spec.
- `analyze-deps` – inspect dependency graph health.
- `apply-modifications` – apply bulk structural edits from a diff/plan.
- `parse-review` – transform review feedback into structured actions.
- `update-frontmatter` – mutate top-level metadata blocks.
- `add-task` / `remove-task` – CRUD operations for tasks within specs.
- `add-assumption` / `list-assumptions` – manage assumption blocks.
- `add-revision` – append revision history entries.
- `add-verification` / `execute-verify` / `format-verification-summary` – manage verification artifacts attached to tasks/specs.
- `update-estimate` – tweak effort/time estimates per task.
- `sync-metadata` – push/pull spec metadata across stores.
- `update-task-metadata` – mutate per-task metadata payloads.
- `create-task-commit` – generate Git commits for task-scoped changes.
- `bulk-journal` – add multiple journal entries in one shot.

### Task Planning & Execution Utilities
- `prepare-task` – build execution context, dependencies, and notes for a task.
- `format-plan` – pretty-print task plans for sharing.
- `query-tasks` – advanced filtering/query over task sets.
- `list-phases` – enumerate phases in a spec.
- `check-complete` – verify completion readiness for a phase/spec.
- `phase-time` – summarize timelines per phase.
- `reconcile-state` – compare file system vs. spec state for drift.
- `check-journaling` – ensure required journaling steps occurred.
- `time-report` – summarize time tracking metrics.
- `status-report` – build structured progress summaries.
- `audit-spec` – run higher-level audits beyond basic validation.

### Lifecycle & Dependency Extras
- `add-assumption`, `list-assumptions`, `add-revision`, `add-verification`, `update-estimate` – richer lifecycle hooks not surfaced in adapters today.
- `add-task` / `remove-task` – lifecycle ops for tasks (distinct from status changes we already support).

### Validation, Reporting & Analytics
- `report` – produce human-readable validation/analysis reports.
- `stats` – aggregate analytics beyond `spec-stats` (e.g., repo-wide metrics).
- `analyze` / `analyze-deps` – deeper data science style insights (if not covered above).

### Collaboration, Review & PR Workflow (LLM-Powered)
These operations require LLM integration - users configure their provider via MCP config:

- `review` – interactive or automated spec review sessions.
- `list-plan-review-tools` – enumerate review toolchains.
- `list-review-tools` – list available review pipelines (non-plan-focused).
- `create-pr` – scaffold GitHub pull requests with SDD context (AI-enhanced descriptions).

### Documentation, Rendering & Skills (LLM-Powered)
- `doc` – generate human-facing documentation bundles.
- `render` – produce rendered artifacts (markdown, with optional AI insights).
- `llm-doc-gen` – LLM-powered doc generation suite.
- `fidelity-review` – specialized documentation fidelity checks (code-to-spec comparison).

### Miscellaneous Tooling
- `cache` – manage SDD CLI cache entries.
- `schema` – emit JSON schemas for specs/tasks.

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
To avoid another retrofit cycle, every new MCP operation or adapter should:

1. **Use the shared helpers** once `foundry_mcp.core.responses` lands (`success_response` / `error_response`). No tool should shape ad-hoc dicts.
2. **Emit the v2 envelope** (`success`, `data`, `error`, `meta.version=response-v2`) even when a tool currently returns no payload—set `data={}` and `error=null`.
3. **Document `meta` usage** when adding pagination, warnings, or telemetry flags so downstream clients can rely on consistent naming.
4. **Add regression tests** (or extend `tests/integration/test_mcp_tools.py`) that assert the response envelope before marking an operation “covered.”
5. **Note deviations in specs**: if an operation legitimately streams multiple payloads, capture the exception rationale in its spec so future refactors don’t “fix” it.

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
