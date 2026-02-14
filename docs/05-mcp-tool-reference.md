# MCP Tool Reference

foundry-mcp exposes 14 unified tools with an `action` parameter that switches behavior. The authoritative schemas live in `mcp/capabilities_manifest.json`.

> **Note:** The `pr`, `code`, and `test` tools have been removed from the MCP surface. Use CLI alternatives instead:
> - `pr`: Use `gh` CLI or git commands directly
> - `code`: Use LSP-integrated editors (Cursor, VS Code)
> - `test`: Use `pytest` directly or `foundry-cli test run`

## Tool Overview

| Tool | Description | Actions |
|------|-------------|---------|
| `health` | Health checks and diagnostics | `liveness`, `readiness`, `check` |
| `spec` | Spec discovery, validation, analysis | `find`, `get`, `list`, `validate`, `fix`, `stats`, `analyze`, `analyze-deps`, `schema`, `diff`, `history`, `completeness-check`, `duplicate-detection` |
| `task` | Task management and batch operations | `prepare`, `prepare-batch`, `start-batch`, `complete-batch`, `reset-batch`, `session-config`, `next`, `info`, `check-deps`, `start`, `complete`, `update-status`, `block`, `unblock`, `list-blocked`, `add`, `remove`, `update-estimate`, `update-metadata`, `progress`, `list`, `query`, `hierarchy`, `move`, `add-dependency`, `remove-dependency`, `add-requirement`, `metadata-batch` |
| `authoring` | Spec authoring and mutations | `spec-create`, `spec-template`, `spec-update-frontmatter`, `phase-add`, `phase-add-bulk`, `phase-remove`, `phase-move`, `phase-template`, `phase-update-metadata`, `assumption-add`, `assumption-list`, `revision-add`, `spec-find-replace`, `spec-rollback`, `intake-add`, `intake-list`, `intake-dismiss` |
| `lifecycle` | Spec lifecycle transitions | `move`, `activate`, `complete`, `archive`, `state` |
| `plan` | Planning helpers | `create`, `list`, `review` |
| `review` | LLM-assisted review workflows | `spec`, `fidelity`, `parse-feedback`, `list-tools`, `list-plan-tools` |
| `verification` | Verification definition and execution | `add`, `execute` |
| `journal` | Journaling helpers | `add`, `list`, `list-unjournaled` |
| `provider` | LLM provider discovery | `list`, `status`, `execute` |
| `environment` | Workspace setup and verification | `init`, `verify-env`, `verify-toolchain`, `setup`, `detect` |
| `error` | Error collection and cleanup | `list`, `get`, `stats`, `patterns`, `cleanup` |
| `research` | AI-powered research workflows | `chat`, `consensus`, `thinkdeep`, `ideate`, `deep-research`, `deep-research-status`, `deep-research-report`, `deep-research-list`, `deep-research-delete`, `thread-list`, `thread-get`, `thread-delete`, `node-execute`, `node-record`, `node-status`, `node-findings`, `extract` |
| `server` | Tool discovery and capabilities | `tools`, `schema`, `capabilities`, `context`, `llm-status` |

---

## Response Envelope

Every tool returns a standard envelope:

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "meta": {
    "version": "response-v2",
    "request_id": "req_abc123",
    "warnings": [],
    "pagination": { "cursor": "...", "has_more": true }
  }
}
```

See [Response Envelope Guide](concepts/response-envelope.md) for details.

---

## health

Health checks and diagnostics.

### Actions

| Action | Description |
|--------|-------------|
| `liveness` | Basic liveness check |
| `readiness` | Readiness check with dependencies |
| `check` | Full health check with details |

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | Yes | - | Health action to run |
| `include_details` | boolean | No | `true` | Include dependency details (action=check) |

### Example

```json
{"action": "liveness"}
```

**CLI equivalent:** None (MCP-only)

---

## spec

Spec discovery, validation, and analysis.

### Actions

| Action | Description |
|--------|-------------|
| `find` | Find a spec by ID |
| `get` | Get spec with full hierarchy |
| `list` | List specs with optional status filter |
| `validate` | Validate spec structure |
| `fix` | Auto-fix validation issues |
| `stats` | Get spec statistics |
| `analyze` | Analyze spec structure |
| `analyze-deps` | Analyze dependency graph |
| `schema` | Get spec JSON schema |
| `diff` | Diff spec against backup |
| `history` | View spec history/backups |
| `completeness-check` | Check spec completeness score |
| `duplicate-detection` | Detect duplicate tasks |

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | Yes | - | Spec action |
| `spec_id` | string | Varies | - | Target spec ID |
| `workspace` | string | No | - | Workspace path override |
| `status` | string | No | `all` | Filter by status (`all`, `pending`, `active`, `completed`, `archived`) |
| `include_progress` | boolean | No | `true` | Include progress in list results |
| `cursor` | string | No | - | Pagination cursor |
| `limit` | integer | No | - | Max results to return |
| `target` | string | No | - | Comparison target for diff |
| `dry_run` | boolean | No | `false` | Preview changes without saving |
| `auto_fix` | boolean | No | `true` | Auto-fix validation issues |

### Examples

```json
{"action": "list", "status": "active"}
{"action": "validate", "spec_id": "my-feature-spec-001"}
{"action": "diff", "spec_id": "my-spec", "target": "20251227T120000"}
```

**CLI equivalent:** `foundry-cli specs find`, `foundry-cli validate check`

---

## task

Task preparation, mutation, and listing.

### Actions

| Action | Description |
|--------|-------------|
| `prepare` | Get next task with context |
| `prepare-batch` | Find independent tasks for parallel execution |
| `start-batch` | Atomically start multiple tasks |
| `complete-batch` | Complete multiple tasks |
| `reset-batch` | Reset stale in_progress tasks |
| `session-config` | Configure/get session settings |
| `next` | Get next actionable task |
| `info` | Get task details |
| `check-deps` | Check task dependencies |
| `start` | Start a task |
| `complete` | Complete a task |
| `update-status` | Update task status |
| `block` | Mark task as blocked |
| `unblock` | Unblock a task |
| `list-blocked` | List blocked tasks |
| `add` | Add a new task |
| `remove` | Remove a task |
| `update-estimate` | Update time estimate |
| `update-metadata` | Update task metadata |
| `progress` | Get progress summary |
| `list` | List tasks |
| `query` | Query tasks with filters |
| `hierarchy` | Get task hierarchy |
| `move` | Move task to new position/parent |
| `add-dependency` | Add dependency between tasks |
| `remove-dependency` | Remove dependency |
| `add-requirement` | Add requirement to task |
| `metadata-batch` | Batch metadata updates |

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | Yes | - | Task action |
| `spec_id` | string | Varies | - | Target spec ID |
| `task_id` | string | Varies | - | Target task ID |
| `dry_run` | boolean | No | `false` | Preview changes |
| `parent` | string | No | - | Target parent for move |
| `position` | integer | No | - | Target position for move |
| `target_id` | string | No | - | Target task for dependencies |
| `dependency_type` | string | No | `blocks` | Dependency type (`blocks`, `blocked_by`, `depends`) |
| `requirement_type` | string | No | - | Requirement type (`acceptance`, `technical`, `constraint`) |
| `text` | string | No | - | Requirement text |

### Examples

```json
{"action": "prepare", "spec_id": "my-feature-spec-001"}
{"action": "complete", "spec_id": "my-spec", "task_id": "task-1-2", "completion_note": "Done"}
{"action": "move", "spec_id": "my-spec", "task_id": "task-1-3", "parent": "phase-2", "position": 1}
```

**CLI equivalent:** `foundry-cli tasks next`, `foundry-cli tasks complete`

---

## authoring

Spec authoring mutations.

### Actions

| Action | Description |
|--------|-------------|
| `spec-create` | Create a new spec |
| `spec-template` | List/show/apply templates |
| `spec-update-frontmatter` | Update frontmatter field |
| `phase-add` | Add a phase |
| `phase-add-bulk` | Add multiple phases |
| `phase-remove` | Remove a phase |
| `phase-move` | Move phase position |
| `phase-template` | Apply phase template |
| `phase-update-metadata` | Update phase metadata |
| `assumption-add` | Add assumption |
| `assumption-list` | List assumptions |
| `revision-add` | Add revision note |
| `spec-find-replace` | Find/replace in spec |
| `spec-rollback` | Rollback to backup |
| `intake-add` | Add intake item |
| `intake-list` | List intake items |
| `intake-dismiss` | Dismiss intake item |

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | Yes | - | Authoring action |
| `spec_id` | string | Varies | - | Target spec ID |
| `name` | string | Varies | - | Spec name (for spec-create) |
| `template` | string | No | - | Template name |
| `category` | string | No | - | Default task category |
| `key` | string | Varies | - | Frontmatter key |
| `value` | string | Varies | - | Frontmatter value |
| `phase_id` | string | Varies | - | Phase identifier |
| `position` | integer | No | - | Target position (1-based) |
| `find` | string | No | - | Text/regex to find |
| `replace` | string | No | - | Replacement text |
| `scope` | string | No | `all` | Find-replace scope (`all`, `titles`, `descriptions`) |
| `use_regex` | boolean | No | `false` | Treat find as regex |
| `title` | string | Varies | - | Intake item title |
| `priority` | string | No | `p2` | Priority (`p0`-`p4`) |
| `tags` | array | No | `[]` | Tags for intake item |
| `dry_run` | boolean | No | `false` | Preview changes |

### Examples

```json
{"action": "spec-create", "name": "my-new-feature", "template": "feature"}
{"action": "phase-move", "spec_id": "my-spec", "phase_id": "phase-3", "position": 1}
{"action": "intake-add", "title": "Add dark mode", "priority": "p2", "tags": ["ui"]}
```

**CLI equivalent:** `foundry-cli specs create`, `foundry-cli modify`

---

## lifecycle

Spec lifecycle transitions.

### Actions

| Action | Description |
|--------|-------------|
| `move` | Move spec between folders |
| `activate` | Activate spec (pending → active) |
| `complete` | Mark spec as finished successfully (requires 100% progress or `force`) |
| `archive` | Archive spec that won't be pursued (abandoned, superseded, deprioritized) |
| `state` | Get lifecycle state |

### Complete vs Archive

- **complete**: Use when all tasks are done and the spec achieved its goals. Sets status to `completed`.
- **archive**: Use when you decide not to pursue the spec, regardless of progress. Sets status to `archived`.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | Yes | - | Lifecycle action |
| `spec_id` | string | Yes | - | Target spec ID |
| `to_folder` | string | For move | - | Target folder |
| `force` | boolean | No | `false` | Force completion |

### Examples

```json
{"action": "activate", "spec_id": "my-feature-spec"}
{"action": "complete", "spec_id": "my-spec", "force": true}
```

**CLI equivalent:** `foundry-cli lifecycle activate`

---

## review

LLM-assisted review workflows.

### Actions

| Action | Description |
|--------|-------------|
| `spec` | Review a specification |
| `fidelity` | Compare implementation to spec |
| `parse-feedback` | Parse review feedback |
| `list-tools` | List review toolchains |
| `list-plan-tools` | List plan review tools |

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | Yes | - | Review action |
| `spec_id` | string | Varies | - | Target spec ID |
| `review_type` | string | No | `full` | Review type (`quick`, `full`, `security`, `feasibility`) |
| `ai_provider` | string | No | - | AI provider to use |
| `ai_timeout` | number | No | 360 | Consultation timeout |
| `consultation_cache` | boolean | No | `true` | Use consultation cache |
| `dry_run` | boolean | No | `false` | Preview without executing |

### Examples

```json
{"action": "spec", "spec_id": "my-spec", "review_type": "security"}
{"action": "fidelity", "spec_id": "my-spec", "task_id": "task-1-2"}
```

**CLI equivalent:** `foundry-cli review spec`

---

## journal

Journaling add/list helpers.

### Actions

| Action | Description |
|--------|-------------|
| `add` | Add journal entry |
| `list` | List journal entries |
| `list-unjournaled` | List tasks without entries |

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | Yes | - | Journal action |
| `spec_id` | string | Yes | - | Spec ID |
| `task_id` | string | No | - | Task ID filter |
| `content` | string | For add | - | Entry content |
| `title` | string | For add | - | Entry title |
| `entry_type` | string | No | - | Entry type filter |

**CLI equivalent:** `foundry-cli journal`

---

## server

Tool discovery and capabilities.

### Actions

| Action | Description |
|--------|-------------|
| `tools` | List available tools |
| `schema` | Get tool schema |
| `capabilities` | Get server capabilities |
| `context` | Get current context |
| `llm-status` | Get LLM provider status |

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | Yes | - | Server action |
| `tool_name` | string | For schema | - | Tool name |
| `category` | string | No | - | Filter by category |
| `tag` | string | No | - | Filter by tag |
| `include_deprecated` | boolean | No | `false` | Include deprecated |

### Examples

```json
{"action": "tools"}
{"action": "schema", "tool_name": "spec"}
```

---

## Related

- [CLI Command Reference](04-cli-command-reference.md) - CLI equivalents
- [Response Envelope Guide](concepts/response-envelope.md) - Response format
- [Error Codes Reference](reference/error-codes.md) - Error handling
- [Intake Guide](guides/intake.md) - Intake workflow details
