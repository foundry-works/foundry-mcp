# CLI Command Reference

foundry-mcp ships a JSON-first CLI for managing specs and workflows.

## Entry Points

- `foundry-cli` (installed script)
- `python -m foundry_mcp.cli` (module entry)

Use `--help` on any command to see full options:

```bash
foundry-cli --help
foundry-cli specs --help
```

## Output Format

CLI output uses the same response envelope as MCP tools, emitted as JSON to stdout. Use `jq` or your JSON tooling of choice for filtering.

See [Response Envelope Guide](concepts/response-envelope.md) for the standard envelope format.

---

## Command Groups Overview

| Group | Purpose | MCP Equivalent |
|-------|---------|----------------|
| `specs` | Create and inspect specs | `spec`, `authoring` |
| `tasks` | Discover and update tasks | `task` |
| `lifecycle` | Activate, complete, archive specs | `lifecycle` |
| `plan` | Plan creation and review helpers | `plan` |
| `review` | LLM-assisted reviews and tooling | `review` |
| `validate` | Spec validation and diagnostics | `spec` (validate/fix actions) |
| `journal` | Journal entries and summaries | `journal` |
| `session` | Session gating and limits | `task` (session-config) |
| `modify` | Batch and edit helpers | `authoring` |
| `cache` | Cache inspection and management | - |
| `dev` | Developer utilities | - |

---

## specs

Specification management commands.

### specs create

Create a new specification.

```bash
foundry-cli specs create <NAME> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Human-readable name for the specification |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--template` | choice | `empty` | Spec template (`empty` only - use phase templates to add structure) |
| `--category` | choice | `implementation` | Default task category (`investigation`, `implementation`, `refactoring`, `decision`, `research`) |
| `--mission` | string | `""` | Optional mission statement for the spec |

**Example:**

```bash
foundry-cli specs create "Add user authentication" --category implementation
```

**MCP equivalent:** `authoring` tool with `action=spec-create`

---

### specs find

Find all specifications with progress information.

```bash
foundry-cli specs find [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--status`, `-s` | choice | `all` | Filter by status (`active`, `pending`, `completed`, `archived`, `all`) |

**Example:**

```bash
foundry-cli specs find --status active
```

**MCP equivalent:** `spec` tool with `action=list`

---

### specs list-phases

List all phases in a specification with progress.

```bash
foundry-cli specs list-phases <SPEC_ID>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

**Example:**

```bash
foundry-cli specs list-phases my-feature-2025-01-15-001
```

---

### specs query-tasks

Query tasks in a specification with filters.

```bash
foundry-cli specs query-tasks <SPEC_ID> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--status`, `-s` | string | - | Filter by status (`pending`, `in_progress`, `completed`, `blocked`) |
| `--parent`, `-p` | string | - | Filter by parent node ID (e.g., `phase-1`) |

**Example:**

```bash
foundry-cli specs query-tasks my-spec --status pending --parent phase-2
```

**MCP equivalent:** `task` tool with `action=list` or `action=query`

---

### specs list-blockers

List all blocked tasks in a specification.

```bash
foundry-cli specs list-blockers <SPEC_ID>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

**Example:**

```bash
foundry-cli specs list-blockers my-spec
```

**MCP equivalent:** `task` tool with `action=list-blocked`

---

### specs template

List or show spec templates.

```bash
foundry-cli specs template <ACTION> [TEMPLATE_NAME]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `ACTION` | Yes | `list` (show all templates) or `show` (show template details) |
| `TEMPLATE_NAME` | For `show` | Template name to display |

**Example:**

```bash
foundry-cli specs template list
foundry-cli specs template show empty
```

**MCP equivalent:** `authoring` tool with `action=spec-template`

---

### specs analyze

Analyze specs directory structure and health.

```bash
foundry-cli specs analyze [DIRECTORY]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `DIRECTORY` | No | Path to analyze (defaults to current directory) |

**Example:**

```bash
foundry-cli specs analyze /path/to/project
```

**MCP equivalent:** `spec` tool with `action=analyze`

---

### specs schema

Export the SDD spec JSON schema.

```bash
foundry-cli specs schema
```

Returns the complete JSON schema for SDD specification files, useful for validation, IDE integration, and agent understanding.

**MCP equivalent:** `spec` tool with `action=schema`

---

## tasks

Task management commands.

### tasks next

Find the next actionable task in a specification.

```bash
foundry-cli tasks next <SPEC_ID>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

**Example:**

```bash
foundry-cli tasks next my-feature-spec
```

**MCP equivalent:** `task` tool with `action=next`

---

### tasks prepare

Prepare complete context for task implementation.

```bash
foundry-cli tasks prepare <SPEC_ID> [TASK_ID]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |
| `TASK_ID` | No | Task identifier (auto-discovers next task if not provided) |

**Example:**

```bash
foundry-cli tasks prepare my-spec task-1-2
```

**MCP equivalent:** `task` tool with `action=prepare`

---

### tasks info

Get detailed information about a specific task.

```bash
foundry-cli tasks info <SPEC_ID> <TASK_ID> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |
| `TASK_ID` | Yes | The task identifier |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--include-context/--no-context` | flag | `--include-context` | Include task context (phase, parent, siblings) |

**Example:**

```bash
foundry-cli tasks info my-spec task-1-2 --include-context
```

**MCP equivalent:** `task` tool with `action=info`

---

### tasks update-status

Update a task's status.

```bash
foundry-cli tasks update-status <SPEC_ID> <TASK_ID> <STATUS> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |
| `TASK_ID` | Yes | The task identifier |
| `STATUS` | Yes | New status (`pending`, `in_progress`, `completed`, `blocked`) |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--note`, `-n` | string | - | Optional note about the status change |

**Example:**

```bash
foundry-cli tasks update-status my-spec task-1-2 in_progress --note "Starting implementation"
```

**MCP equivalent:** `task` tool with `action=update-status`

---

### tasks complete

Mark a task as completed with auto-journaling.

```bash
foundry-cli tasks complete <SPEC_ID> <TASK_ID> --note <NOTE>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |
| `TASK_ID` | Yes | The task identifier |

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--note`, `-n` | string | Yes | Completion note describing what was accomplished |

**Example:**

```bash
foundry-cli tasks complete my-spec task-1-2 --note "Implemented the authentication handler"
```

**MCP equivalent:** `task` tool with `action=complete`

---

### tasks block

Mark a task as blocked.

```bash
foundry-cli tasks block <SPEC_ID> <TASK_ID> --reason <REASON> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |
| `TASK_ID` | Yes | The task identifier |

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--reason`, `-r` | string | Yes | - | Description of the blocker |
| `--type`, `-t` | choice | No | `dependency` | Blocker type (`dependency`, `technical`, `resource`, `decision`) |
| `--ticket` | string | No | - | Optional ticket/issue reference |

**Example:**

```bash
foundry-cli tasks block my-spec task-1-2 --reason "Waiting for API design review" --type decision
```

**MCP equivalent:** `task` tool with `action=block`

---

### tasks unblock

Unblock a task.

```bash
foundry-cli tasks unblock <SPEC_ID> <TASK_ID> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |
| `TASK_ID` | Yes | The task identifier |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--resolution`, `-r` | string | - | Description of how blocker was resolved |
| `--status`, `-s` | choice | `pending` | Status after unblocking (`pending`, `in_progress`) |

**Example:**

```bash
foundry-cli tasks unblock my-spec task-1-2 --resolution "API design approved" --status in_progress
```

**MCP equivalent:** `task` tool with `action=unblock`

---

### tasks check-complete

Check if a task can be marked as complete.

```bash
foundry-cli tasks check-complete <SPEC_ID> <TASK_ID>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |
| `TASK_ID` | Yes | The task identifier |

Returns whether the task can be completed and any blockers preventing completion (dependencies, child tasks, blocked status).

**Example:**

```bash
foundry-cli tasks check-complete my-spec task-1-2
```

**MCP equivalent:** `task` tool with `action=check-deps`

---

## lifecycle

Spec lifecycle management commands.

### lifecycle activate

Activate a specification (move from pending to active).

```bash
foundry-cli lifecycle activate <SPEC_ID>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

**Example:**

```bash
foundry-cli lifecycle activate my-feature-spec
```

**MCP equivalent:** `lifecycle` tool with `action=activate`

---

### lifecycle complete

Mark a specification as completed.

```bash
foundry-cli lifecycle complete <SPEC_ID> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--force`, `-f` | flag | false | Force completion even with incomplete tasks |

**Example:**

```bash
foundry-cli lifecycle complete my-spec
foundry-cli lifecycle complete my-spec --force
```

**MCP equivalent:** `lifecycle` tool with `action=complete`

---

### lifecycle archive

Archive a specification.

```bash
foundry-cli lifecycle archive <SPEC_ID>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

**Example:**

```bash
foundry-cli lifecycle archive old-feature-spec
```

**MCP equivalent:** `lifecycle` tool with `action=archive`

---

### lifecycle move

Move a specification between status folders.

```bash
foundry-cli lifecycle move <SPEC_ID> <TO_FOLDER>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |
| `TO_FOLDER` | Yes | Target folder (`pending`, `active`, `completed`, `archived`) |

**Example:**

```bash
foundry-cli lifecycle move my-spec active
```

**MCP equivalent:** `lifecycle` tool with `action=move`

---

### lifecycle state

Get the current lifecycle state of a specification.

```bash
foundry-cli lifecycle state <SPEC_ID>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

Returns folder, status, progress, task counts, and available transitions.

**Example:**

```bash
foundry-cli lifecycle state my-spec
```

**MCP equivalent:** `lifecycle` tool with `action=state`

---

## plan

Markdown plan review commands.

### plan create

Create a new markdown implementation plan.

```bash
foundry-cli plan create <NAME> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Human-readable plan name |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--template` | choice | `detailed` | Plan template (`simple`, `detailed`) |

Creates a plan file in `specs/.plans/` with the specified template.

**Example:**

```bash
foundry-cli plan create "Add user authentication"
foundry-cli plan create "Refactor database layer" --template simple
```

**MCP equivalent:** `plan` tool with `action=create`

---

### plan list

List all markdown implementation plans.

```bash
foundry-cli plan list
```

Lists plans from `specs/.plans/` directory with review status.

**Example:**

```bash
foundry-cli plan list
```

**MCP equivalent:** `plan` tool with `action=list`

---

### plan review

Review a markdown implementation plan with AI feedback.

```bash
foundry-cli plan review <PLAN_PATH> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `PLAN_PATH` | Yes | Path to the markdown plan file |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--type` | choice | `full` | Review type (`quick`, `full`, `security`, `feasibility`) |
| `--ai-provider` | string | - | Explicit AI provider selection (e.g., `gemini`, `cursor-agent`) |
| `--ai-timeout` | float | 360 | AI consultation timeout in seconds |
| `--no-consultation-cache` | flag | false | Bypass AI consultation cache |
| `--dry-run` | flag | false | Show what would be reviewed without executing |

Writes review output to `specs/.plan-reviews/<plan-name>-<review-type>.md`.

**Example:**

```bash
foundry-cli plan review ./PLAN.md
foundry-cli plan review ./PLAN.md --type security
foundry-cli plan review ./PLAN.md --ai-provider gemini
```

**MCP equivalent:** `plan` tool with `action=review`

---

## review

Spec review and fidelity checking commands.

### review spec

Run a structural or AI-powered review on a specification.

```bash
foundry-cli review spec <SPEC_ID> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--type` | choice | `full` | Review type (`quick`, `full`, `security`, `feasibility`) |
| `--tools` | string | - | Comma-separated list of review tools (LLM types only) |
| `--model` | string | - | LLM model to use for review |
| `--ai-provider` | string | - | Explicit AI provider selection |
| `--ai-timeout` | float | 360 | AI consultation timeout in seconds |
| `--no-consultation-cache` | flag | false | Bypass AI consultation cache |
| `--dry-run` | flag | false | Show what would be reviewed without executing |

**Example:**

```bash
foundry-cli review spec my-spec
foundry-cli review spec my-spec --type quick
foundry-cli review spec my-spec --type security --ai-provider gemini
```

**MCP equivalent:** `review` tool with `action=spec`

---

### review fidelity

Compare implementation against specification.

```bash
foundry-cli review fidelity <SPEC_ID> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--task` | string | - | Review specific task implementation |
| `--phase` | string | - | Review entire phase implementation |
| `--files` | string | - | Review specific file(s) only (can specify multiple) |
| `--incremental` | flag | false | Only review changed files since last run |
| `--base-branch` | string | `main` | Base branch for git diff |
| `--ai-provider` | string | - | Explicit AI provider selection |
| `--ai-timeout` | float | 360 | AI consultation timeout in seconds |
| `--no-consultation-cache` | flag | false | Bypass AI consultation cache |

**Example:**

```bash
foundry-cli review fidelity my-spec
foundry-cli review fidelity my-spec --task task-1-2
foundry-cli review fidelity my-spec --phase phase-1 --incremental
```

**MCP equivalent:** `review` tool with `action=fidelity`

---

### review tools

List native and external review toolchains.

```bash
foundry-cli review tools
```

**MCP equivalent:** `review` tool with `action=list-tools`

---

### review plan-tools

List available plan review toolchains.

```bash
foundry-cli review plan-tools
```

**MCP equivalent:** `review` tool with `action=list-plan-tools`

---

## validate

Spec validation and fix commands.

### validate check

Validate a specification and report diagnostics.

```bash
foundry-cli validate check <SPEC_ID>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

**Example:**

```bash
foundry-cli validate check my-spec
```

**MCP equivalent:** `spec` tool with `action=validate`

---

### validate fix

Apply auto-fixes to a specification.

```bash
foundry-cli validate fix <SPEC_ID> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dry-run` | flag | false | Preview fixes without applying |
| `--no-backup` | flag | false | Skip creating backup file |

**Example:**

```bash
foundry-cli validate fix my-spec
foundry-cli validate fix my-spec --dry-run
```

**MCP equivalent:** `spec` tool with `action=fix`

---

### validate stats

Get statistics for a specification.

```bash
foundry-cli validate stats <SPEC_ID>
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

Returns task counts, progress, verification coverage, and file size.

**Example:**

```bash
foundry-cli validate stats my-spec
```

**MCP equivalent:** `spec` tool with `action=stats`

---

### validate report

Generate a comprehensive report for a specification.

```bash
foundry-cli validate report <SPEC_ID> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--sections`, `-s` | string | `all` | Sections to include: `validation`, `stats`, `health`, `all` |

**Example:**

```bash
foundry-cli validate report my-spec
foundry-cli validate report my-spec --sections validation,stats
```

---

### validate analyze-deps

Analyze dependency graph health for a specification.

```bash
foundry-cli validate analyze-deps <SPEC_ID> [OPTIONS]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `SPEC_ID` | Yes | The specification identifier |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--bottleneck-threshold`, `-t` | int | 3 | Minimum tasks blocked to flag as bottleneck |
| `--limit` | int | 100 | Maximum items to return per section |

Identifies blocking tasks, bottlenecks, circular dependencies, and the critical path.

**Example:**

```bash
foundry-cli validate analyze-deps my-spec
foundry-cli validate analyze-deps my-spec --bottleneck-threshold 5
```

**MCP equivalent:** `spec` tool with `action=analyze-deps`

---

## Top-Level Aliases

These commands are available at the root level for convenience:

### validate (top-level)

Validate a specification and optionally apply fixes.

```bash
foundry-cli validate <SPEC_ID> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--fix` | flag | false | Auto-fix issues after validation |
| `--dry-run` | flag | false | Preview fixes without applying (requires `--fix`) |
| `--preview` | flag | false | Show summary only (counts and issue codes) |
| `--diff` | flag | false | Show unified diff of changes (requires `--fix`) |
| `--select` | string | - | Only fix selected issue codes (comma-separated) |

**Example:**

```bash
foundry-cli validate my-spec
foundry-cli validate my-spec --fix --dry-run
foundry-cli validate my-spec --fix --select "MISSING_METADATA,INVALID_STATUS"
```

---

### fix (top-level)

Apply auto-fixes to a specification.

```bash
foundry-cli fix <SPEC_ID> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dry-run` | flag | false | Preview fixes without applying |
| `--no-backup` | flag | false | Skip creating backup file |
| `--diff` | flag | false | Show unified diff of changes |
| `--select` | string | - | Only fix selected issue codes (comma-separated) |

**Example:**

```bash
foundry-cli fix my-spec
foundry-cli fix my-spec --dry-run --diff
```

---

## Global Options

These options are available for all commands:

| Option | Type | Description |
|--------|------|-------------|
| `--specs-dir` | path | Override specs directory path |
| `--help` | flag | Show command help |

Environment variables:

| Variable | Description |
|----------|-------------|
| `FOUNDRY_MCP_SPECS_DIR` | Default specs directory path |
| `SDD_SPECS_DIR` | Alternative specs directory variable |
| `FOUNDRY_MCP_LOG_LEVEL` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## Related

- [MCP Tool Reference](05-mcp-tool-reference.md) - MCP tool equivalents
- [Configuration](06-configuration.md) - Environment and config options
- [Troubleshooting](07-troubleshooting.md) - Common issues and fixes
