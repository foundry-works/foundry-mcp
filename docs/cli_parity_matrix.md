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
