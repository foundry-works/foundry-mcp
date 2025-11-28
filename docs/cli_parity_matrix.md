# SDD CLI to MCP Tools Parity Matrix

This document captures the mapping between `sdd` CLI commands (from claude_skills) and foundry-mcp MCP tools.

## CLI Registry Source

The CLI commands are registered via `claude_skills.cli.sdd.registry` which imports from:

- `claude_skills.sdd_next.cli` - Task discovery and preparation
- `claude_skills.sdd_update.cli` - Status and journal updates
- `claude_skills.sdd_validate.cli` - Spec validation
- `claude_skills.sdd_plan.cli` - Planning workflows
- `claude_skills.sdd_plan_review.cli` - AI-powered plan reviews
- `claude_skills.sdd_pr.cli` - PR creation workflows
- `claude_skills.context_tracker.cli` - Context monitoring
- `claude_skills.sdd_spec_mod.cli` - Spec modifications
- `claude_skills.common.cache.cli` - Cache management
- `claude_skills.sdd_render.cli` (optional) - Markdown rendering
- `claude_skills.sdd_fidelity_review.cli` (optional) - Fidelity reviews

## Parity Matrix

### Core Spec Operations

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `find-specs` | Find specs directory | `spec-list`, `spec-find` | Implemented |
| `list-specs` | List specification files | `spec-list`, `spec-list-by-folder` | Implemented |
| `progress` | Show overall progress | `task-progress`, `spec-render-progress` | Implemented |
| `list-phases` | List all phases with progress | `spec-get-hierarchy` | Partial |
| `spec-stats` | Show spec file statistics | `spec-stats` | Implemented |

### Task Operations

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `next-task` | Find next actionable task | `task-next` | Implemented |
| `prepare-task` | Prepare task for implementation | `task-prepare` | Implemented |
| `task-info` | Get task information | `task-info`, `task-get` | Implemented |
| `get-task` | Get detailed task information | `task-get` | Implemented |
| `check-deps` | Check task dependencies | `task-check-deps` | Implemented |
| `query-tasks` | Query and filter tasks | `task-query` | Implemented |
| `update-status` | Update task status | `task-update-status` | Implemented |
| `mark-blocked` | Mark task as blocked | `task-block` | Implemented |
| `unblock-task` | Unblock a task | `task-unblock` | Implemented |
| `list-blockers` | List all blocked tasks | `task-list-blocked` | Implemented |
| `complete-task` | Complete task with journaling | `task-complete` | Implemented |
| `add-task` | Add a new task to hierarchy | `task-add` | Implemented |
| `remove-task` | Remove a task from hierarchy | `task-remove` | Implemented |
| `update-estimate` | Update task estimate | `task-update-estimate` | Implemented |
| `update-task-metadata` | Update task metadata fields | `task-update-metadata` | Implemented |
| `task-start` | Mark task as in_progress | `task-start` | Implemented |

### Validation & Fixing

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `validate` | Validate JSON spec file | `spec-validate` | Implemented |
| `validate-spec` | Validate spec file (alias) | `spec-validate` | Implemented |
| `fix` | Auto-fix validation issues | `spec-fix` | Implemented |
| `validate-paths` | Validate and normalize paths | `spec-validate-paths` | Implemented |
| `find-circular-deps` | Find circular dependencies | `spec-detect-cycles` | Implemented |
| `analyze-deps` | Analyze dependencies | `spec-analyze-deps` | Implemented |

### Journal Operations

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `add-journal` | Add journal entry | `journal-add` | Implemented |
| `get-journal` | Get journal entries | `journal-list` | Implemented |
| `check-journaling` | Check for unjournaled tasks | `journal-list-unjournaled` | Implemented |
| `bulk-journal` | Bulk journal completed tasks | - | Not Implemented |

### Lifecycle Operations

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `move-spec` | Move spec to another folder | `spec-lifecycle-move` | Implemented |
| `complete-spec` | Mark spec as completed | `spec-lifecycle-complete` | Implemented |
| `activate-spec` | Activate a pending spec | `spec-lifecycle-activate` | Implemented |
| `check-complete` | Check if ready to complete | - | Not Implemented |

### Authoring Operations

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `create` | Create new specification | `spec-create` | Implemented |
| `template` | Manage spec templates | `spec-template` | Implemented |
| `add-revision` | Add revision metadata entry | `revision-add` | Implemented |
| `add-assumption` | Add assumption to spec | `assumption-add` | Implemented |
| `list-assumptions` | List assumptions | `assumption-list` | Implemented |
| `update-frontmatter` | Update spec frontmatter | `spec-update-frontmatter` | Implemented |
| `apply-modifications` | Apply batch modifications | `spec-apply-plan` | Implemented |

### Verification Operations

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `add-verification` | Add verification result | `verification-add` | Implemented |
| `execute-verify` | Execute verification task | `verification-execute` | Implemented |
| `format-verification-summary` | Format verification results | `verification-format-summary` | Implemented |

### Documentation & Rendering

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `render` | Render spec to markdown | `spec-render` | Implemented |
| `doc` | Documentation commands | `spec-doc`, `spec-doc-llm` | Implemented |
| `llm-doc-gen` | Generate docs using LLMs | `spec-doc-llm` | Implemented |

### Code Documentation Query

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `doc scope` | Query code documentation | `code-find-class`, `code-find-function` | Partial |
| `doc stats` | Get documentation stats | `doc-stats` | Implemented |
| - | Trace function calls | `code-trace-calls` | Implemented |
| - | Impact analysis | `code-impact-analysis` | Implemented |
| - | Get function callers | `code-get-callers` | Implemented |
| - | Get function callees | `code-get-callees` | Implemented |

### Testing

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `test run` | Run tests | `test-run` | Implemented |
| `test discover` | Discover tests | `test-discover` | Implemented |
| - | Get test presets | `test-presets` | Implemented |
| - | Quick test run | `test-run-quick` | Implemented |
| - | Unit test run | `test-run-unit` | Implemented |

### Review Operations

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `review` | Review spec with AI | `spec-review` | Implemented |
| `list-review-tools` | List AI review tools | `review-list-tools` | Implemented |
| `list-plan-review-tools` | List plan review tools | `review-list-plan-tools` | Implemented |
| `fidelity-review` | Review implementation fidelity | `spec-review-fidelity` | Implemented |
| `parse-review` | Parse review report | `review-parse-feedback` | Implemented |

### PR Workflow

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `create-pr` | Create AI-powered PR | `pr-create-with-spec` | Implemented |
| - | Get PR spec context | `pr-get-spec-context` | Implemented |

### Environment & Setup

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `verify-tools` | Verify required tools | `sdd-verify-toolchain` | Implemented |
| `init-env` | Initialize dev environment | `sdd-init-workspace` | Implemented |
| `detect-project` | Detect project type | `sdd-detect-topology` | Implemented |
| `check-environment` | Check environmental requirements | `sdd-verify-environment` | Implemented |

### Analysis & Reports

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `analyze` | Analyze codebase | `spec-analyze` | Implemented |
| `report` | Generate validation report | `spec-report` | Implemented |
| `stats` | Show spec statistics | `spec-stats`, `spec-report-summary` | Implemented |
| `time-report` | Generate time tracking report | - | Not Implemented |
| `status-report` | Get status report | `spec-report-summary` | Partial |
| `audit-spec` | Deep audit of JSON spec | - | Not Implemented |
| `phase-time` | Calculate time breakdown | - | Not Implemented |

### Utilities

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `find-pattern` | Find files matching pattern | `spec-find-patterns` | Implemented |
| `find-related-files` | Find related files | `spec-find-related-files` | Implemented |
| `find-tests` | Find test files | - | Not Implemented |
| `format-plan` | Format execution plan | - | Not Implemented |
| `reconcile-state` | Reconcile spec inconsistencies | - | Not Implemented |
| `sync-metadata` | Synchronize spec metadata | `spec-sync-metadata` | Implemented |

### Context & Session

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `context` | Monitor token/context usage | - | Not Implemented (CLI-specific) |
| `session-marker` | Generate session marker | - | Not Implemented (CLI-specific) |
| `get-work-mode` | Get configured work mode | - | Not Implemented (CLI-specific) |

### Cache Management

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `cache` | Manage AI consultation cache | `sdd-cache-manage` | Implemented |

### Schema & Discovery

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `schema` | Get SDD spec JSON schema | `spec-schema-export` | Implemented |
| - | List available tools | `tool-list` | Implemented |
| - | Get tool schema | `tool-get-schema` | Implemented |
| - | List tool categories | `tool-list-categories` | Implemented |
| - | Get capabilities | `capability-get` | Implemented |
| - | Negotiate capabilities | `capability-negotiate` | Implemented |
| - | Get server context | `get-server-context` | Implemented |
| - | Get LLM status | `get-llm-status` | Implemented |
| - | Server capabilities | `sdd-server-capabilities` | Implemented |

### Git Integration

| CLI Command | Description | MCP Tool | Status |
|-------------|-------------|----------|--------|
| `create-task-commit` | Create commit for task | - | Not Implemented |

## Overlap Classification

This section classifies CLI commands by their MCP adapter status.

### Already Wrapped (MCP Tool Exists)

These CLI commands have corresponding MCP tools and are fully functional:

**Core Operations:**
- `find-specs` → `spec-list`, `spec-find`
- `list-specs` → `spec-list`, `spec-list-by-folder`
- `progress` → `task-progress`, `spec-render-progress`
- `spec-stats` → `spec-stats`

**Task Operations:**
- `next-task` → `task-next`
- `prepare-task` → `task-prepare`
- `task-info` → `task-info`, `task-get`
- `get-task` → `task-get`
- `check-deps` → `task-check-deps`
- `query-tasks` → `task-query`
- `update-status` → `task-update-status`
- `mark-blocked` → `task-block`
- `unblock-task` → `task-unblock`
- `list-blockers` → `task-list-blocked`
- `complete-task` → `task-complete`
- `add-task` → `task-add`
- `remove-task` → `task-remove`
- `update-estimate` → `task-update-estimate`
- `update-task-metadata` → `task-update-metadata`

**Validation:**
- `validate` → `spec-validate`
- `fix` → `spec-fix`
- `validate-paths` → `spec-validate-paths`
- `find-circular-deps` → `spec-detect-cycles`
- `analyze-deps` → `spec-analyze-deps`

**Journal:**
- `add-journal` → `journal-add`
- `get-journal` → `journal-list`
- `check-journaling` → `journal-list-unjournaled`

**Lifecycle:**
- `move-spec` → `spec-lifecycle-move`
- `complete-spec` → `spec-lifecycle-complete`
- `activate-spec` → `spec-lifecycle-activate`

**Authoring:**
- `create` → `spec-create`
- `template` → `spec-template`
- `add-revision` → `revision-add`
- `add-assumption` → `assumption-add`
- `list-assumptions` → `assumption-list`
- `update-frontmatter` → `spec-update-frontmatter`
- `apply-modifications` → `spec-apply-plan`

**Verification:**
- `add-verification` → `verification-add`
- `execute-verify` → `verification-execute`
- `format-verification-summary` → `verification-format-summary`

**Documentation:**
- `render` → `spec-render`
- `doc` → `spec-doc`, `spec-doc-llm`
- `llm-doc-gen` → `spec-doc-llm`
- `doc stats` → `doc-stats`

**Testing:**
- `test run` → `test-run`
- `test discover` → `test-discover`

**Review:**
- `review` → `spec-review`
- `list-review-tools` → `review-list-tools`
- `list-plan-review-tools` → `review-list-plan-tools`
- `fidelity-review` → `spec-review-fidelity`
- `parse-review` → `review-parse-feedback`

**PR Workflow:**
- `create-pr` → `pr-create-with-spec`

**Environment:**
- `verify-tools` → `sdd-verify-toolchain`
- `init-env` → `sdd-init-workspace`
- `detect-project` → `sdd-detect-topology`
- `check-environment` → `sdd-verify-environment`

**Analysis:**
- `analyze` → `spec-analyze`
- `report` → `spec-report`
- `stats` → `spec-stats`, `spec-report-summary`

**Utilities:**
- `find-pattern` → `spec-find-patterns`
- `find-related-files` → `spec-find-related-files`
- `sync-metadata` → `spec-sync-metadata`

**Cache:**
- `cache` → `sdd-cache-manage`

**Schema:**
- `schema` → `spec-schema-export`

### Partial Implementation (Needs Enhancement)

These have MCP tools but with incomplete feature parity:

| CLI Command | MCP Tool | Gap Description |
|-------------|----------|-----------------|
| `list-phases` | `spec-get-hierarchy` | CLI shows progress %, MCP returns raw hierarchy |
| `doc scope` | `code-find-*` | CLI has `--plan`/`--implement` modes not in MCP |
| `status-report` | `spec-report-summary` | CLI has more formatting options |

### Gaps (No MCP Equivalent)

These CLI commands have no MCP tool and need implementation:

| CLI Command | Priority | Rationale |
|-------------|----------|-----------|
| `bulk-journal` | Medium | Useful for batch operations |
| `check-complete` | Medium | Validation before lifecycle transitions |
| `time-report` | Low | Nice-to-have reporting feature |
| `phase-time` | Low | Time tracking per phase |
| `audit-spec` | Low | Deep spec analysis |
| `reconcile-state` | Low | State repair utility |
| `find-tests` | Low | Test discovery utility |
| `format-plan` | Low | Display formatting |
| `create-task-commit` | Medium | Git integration for workflows |

### CLI-Specific (No MCP Needed)

These commands are specific to CLI/Claude Code integration:

| CLI Command | Reason |
|-------------|--------|
| `context` | Monitors Claude Code token usage - requires transcript access |
| `session-marker` | Generates markers for transcript identification |
| `get-work-mode` | Reads local config file for workflow mode |

### MCP-Only Tools (No CLI Equivalent)

These MCP tools extend beyond CLI capabilities:

| MCP Tool | Description |
|----------|-------------|
| `task-start` | Explicit in_progress transition (CLI uses `update-status`) |
| `task-list` | Flat task list view |
| `code-trace-calls` | Function call graph traversal |
| `code-impact-analysis` | Impact analysis for changes |
| `code-get-callers` | Get function callers |
| `code-get-callees` | Get function callees |
| `test-presets` | List test preset configurations |
| `test-run-quick` | Quick test preset |
| `test-run-unit` | Unit test preset |
| `pr-get-spec-context` | Get context for PR generation |
| `tool-list` | MCP tool discovery |
| `tool-get-schema` | MCP tool schema introspection |
| `tool-list-categories` | MCP tool categorization |
| `capability-get` | Server capability query |
| `capability-negotiate` | Capability negotiation |
| `get-server-context` | Server context info |
| `get-llm-status` | LLM provider status |
| `sdd-server-capabilities` | Full server capabilities |
| `spec-lifecycle-state` | Lifecycle state query |
| `spec-validate-fix` | Combined validate + fix |

## Not Implemented in MCP (CLI-Only)

These CLI commands have no MCP equivalent and may not need one:

1. **Context/Session Management** - These are CLI-specific for tracking Claude Code context:
   - `context`
   - `session-marker`
   - `get-work-mode`

2. **Time Tracking** - May be added later:
   - `time-report`
   - `phase-time`

3. **Audit/Reconciliation** - Advanced operations:
   - `audit-spec`
   - `reconcile-state`

4. **Git Operations** - CLI handles git directly:
   - `create-task-commit`

5. **Utility/Format**:
   - `format-plan`
   - `find-tests`
   - `bulk-journal`
   - `check-complete`

## Migration Priority Tiers

Commands and tools are grouped by migration priority to guide implementation efforts.

### Tier 1: Critical (Core Workflows)

Must-have functionality for basic SDD operations. Migration blockers if missing.

**Gap Commands to Implement:**
| Command | Status | Rationale |
|---------|--------|-----------|
| `check-complete` | Not Implemented | Required for lifecycle transitions - blocks spec completion workflow |
| `create-task-commit` | Not Implemented | Essential for git integration in task workflows |

**Tier 1 Command Specifications:**

`check-complete` validates readiness before lifecycle transitions:
- Verifies all tasks in spec/phase have `status: completed`
- Checks no tasks are `blocked` or `in_progress`
- Validates all completed tasks have journal entries (`needs_journaling: false`)
- Returns structured result with blocking issues if not ready
- **Workflow-blocking:** Called before `spec-lifecycle-complete` to prevent premature completion

`create-task-commit` automates git commits during task workflows:
- Creates atomic commit with task context (spec_id, task_id, title)
- Follows configured commit message format from `.claude/sdd_config.json`
- Stages only files modified during task (from task metadata)
- **Workflow-blocking:** Core to the SDD auto-commit workflow when `git.auto_commit: true`

**Partial Commands to Enhance:**
| Command | Current MCP | Enhancement Needed |
|---------|-------------|-------------------|
| `list-phases` | `spec-get-hierarchy` | Add progress % calculation in MCP response |

**Already Wrapped - Critical:**
- All Task Operations (16 commands) - ✅ Full parity
- All Core Spec Operations (5 commands) - ✅ Full parity
- All Lifecycle Operations except `check-complete` - ✅ Full parity
- All Validation Operations (6 commands) - ✅ Full parity

### Tier 2: Important (Documentation & Testing)

Important for documentation, testing, and productivity but not blocking core workflows.

**Gap Commands to Implement:**
| Command | Status | Rationale |
|---------|--------|-----------|
| `bulk-journal` | Not Implemented | Productivity improvement for batch task completion |
| `find-tests` | Not Implemented | Test discovery utility for verification tasks |

**Partial Commands to Enhance:**
| Command | Current MCP | Enhancement Needed |
|---------|-------------|-------------------|
| `doc scope` | `code-find-*` | Add `--plan` and `--implement` mode flags |
| `status-report` | `spec-report-summary` | Add formatting options (markdown, JSON, brief) |

**Already Wrapped - Important:**
- All Documentation Operations (3 commands) - ✅ Full parity
- All Testing Operations (2 commands, 5 MCP tools) - ✅ Full parity
- All Journal Operations except `bulk-journal` - ✅ Full parity
- All Verification Operations (3 commands) - ✅ Full parity
- All Review Operations (5 commands) - ✅ Full parity

### Tier 3: Advanced (AI Rendering & Analysis)

Nice-to-have features for advanced users. Can be deferred.

**Gap Commands to Implement:**
| Command | Status | Rationale |
|---------|--------|-----------|
| `time-report` | Not Implemented | Time tracking report generation |
| `phase-time` | Not Implemented | Time breakdown per phase |
| `audit-spec` | Not Implemented | Deep spec analysis |
| `reconcile-state` | Not Implemented | State repair utility |
| `format-plan` | Not Implemented | Display formatting for plans |

**Already Wrapped - Advanced:**
- All Schema/Discovery Operations (8 MCP tools) - ✅ Full parity
- All Code Query Operations (6 MCP tools) - ✅ Full parity
- All PR Workflow Operations (2 MCP tools) - ✅ Full parity
- Cache Management - ✅ Full parity

### CLI-Specific (No Migration Needed)

These commands are specific to Claude Code integration and don't need MCP equivalents:

| Command | Reason |
|---------|--------|
| `context` | Requires Claude Code transcript access |
| `session-marker` | Claude Code session identification |
| `get-work-mode` | Local config file access |

### Migration Effort Summary

| Tier | Gap Commands | Partial Enhancements | Total Effort |
|------|--------------|---------------------|--------------|
| Tier 1 (Critical) | 2 | 1 | 3 items |
| Tier 2 (Important) | 2 | 2 | 4 items |
| Tier 3 (Advanced) | 5 | 0 | 5 items |
| CLI-Specific | 0 | 0 | N/A |
| **Total** | **9** | **3** | **12 items** |

### Recommended Implementation Order

1. **Phase 1 - Critical Gaps** (Tier 1):
   - `check-complete` → New MCP tool
   - `create-task-commit` → New MCP tool
   - `list-phases` enhancement → Update `spec-get-hierarchy` or add `spec-list-phases`

2. **Phase 2 - Important Gaps** (Tier 2):
   - `bulk-journal` → New MCP tool
   - `find-tests` → New MCP tool
   - `doc scope` enhancement → Add mode flags to `code-find-*` or new tool
   - `status-report` enhancement → Update `spec-report-summary`

3. **Phase 3 - Advanced Gaps** (Tier 3, optional):
   - Time tracking tools
   - Audit/reconciliation tools
   - Format utilities

## Summary

| Category | Total CLI Commands | MCP Implemented | Coverage |
|----------|-------------------|-----------------|----------|
| Core Spec | 5 | 5 | 100% |
| Tasks | 16 | 16 | 100% |
| Validation | 6 | 6 | 100% |
| Journal | 4 | 3 | 75% |
| Lifecycle | 4 | 3 | 75% |
| Authoring | 6 | 6 | 100% |
| Verification | 3 | 3 | 100% |
| Documentation | 3 | 3 | 100% |
| Code Query | 2 | 6 | 300%+ |
| Testing | 2 | 5 | 250%+ |
| Review | 5 | 5 | 100% |
| PR Workflow | 1 | 2 | 200% |
| Environment | 4 | 4 | 100% |
| Analysis | 6 | 4 | 67% |
| Utilities | 6 | 3 | 50% |
| Context/Session | 3 | 0 | 0% (CLI-specific) |
| Cache | 1 | 1 | 100% |
| Schema/Discovery | 1 | 8 | 800%+ |
| Git | 1 | 0 | 0% |

**Overall Coverage: ~85%** (excluding CLI-specific commands)

The MCP implementation has excellent parity with the CLI, and in some areas (testing, code query, discovery) provides more granular tools than the CLI.
