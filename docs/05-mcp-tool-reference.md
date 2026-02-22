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
| `task` | Task management and batch operations | `prepare`, `prepare-batch`, `start-batch`, `complete-batch`, `reset-batch`, `session-config`, `session`, `session-step`, `session-events`, `next`, `info`, `check-deps`, `start`, `complete`, `update-status`, `block`, `unblock`, `list-blocked`, `add`, `remove`, `update-estimate`, `update-metadata`, `progress`, `list`, `query`, `hierarchy`, `move`, `add-dependency`, `remove-dependency`, `add-requirement`, `metadata-batch`, `fix-verification-types`, `gate-waiver` |
| `authoring` | Spec authoring and mutations | `spec-create`, `spec-template`, `spec-update-frontmatter`, `phase-add`, `phase-add-bulk`, `phase-remove`, `phase-move`, `phase-update-metadata`, `assumption-add`, `assumption-list`, `revision-add`, `constraint-add`, `constraint-list`, `risk-add`, `risk-list`, `question-add`, `question-list`, `success-criterion-add`, `success-criteria-list`, `spec-find-replace`, `spec-rollback` |
| `lifecycle` | Spec lifecycle transitions | `move`, `activate`, `complete`, `archive`, `state` |
| `plan` | Planning helpers | `create`, `list`, `review` |
| `review` | LLM-assisted review workflows | `spec`, `fidelity`, `fidelity-gate`, `parse-feedback`, `list-tools`, `list-plan-tools` |
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
| `session` | Canonical autonomous session lifecycle entrypoint (requires `command`) |
| `session-step` | Canonical autonomous session-step entrypoint (requires `command`) |
| `session-events` | Journal-backed event feed for one autonomous session (cursor paginated) |
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
| `fix-verification-types` | Repair invalid verification task types |
| `gate-waiver` | Privileged required-gate waiver (maintainer only) |

Canonical session commands:
`start`, `status`, `pause`, `resume`, `rebase`, `end`, `list`, `reset`

Canonical session-step commands:
`next`, `report`, `replay`, `heartbeat`

Loop supervisor contract (`session-step-next`, `session-step-report`, `session-step-replay`):
- Continue unattended loops only when `data.loop_signal == "phase_complete"`.
- Treat every other non-null signal (`spec_complete`, `paused_needs_attention`, `failed`, `blocked_runtime`) as stop-and-escalate.
- Use `data.recommended_actions` for machine-readable remediation guidance.

Legacy compatibility:
Legacy action names such as `session-start` and `session-step-next` are still accepted, but responses include `meta.deprecated` metadata with a canonical replacement.

Deprecation timeline:
- Legacy action names are scheduled for removal after **3 months or 2 minor releases (whichever is later)**.
- Legacy responses include:
  - `meta.deprecated.action`
  - `meta.deprecated.replacement`
  - `meta.deprecated.removal_target`
- The server also emits `WARN` logs for legacy action invocations to support migration monitoring.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | Yes | - | Task action |
| `command` | string | Varies | - | Required when `action` is `session` or `session-step` |
| `spec_id` | string | Varies | - | Target spec ID |
| `task_id` | string | Varies | - | Target task ID |
| `session_id` | string | Varies | - | Session ID for session/session-step operations |
| `cursor` | string | No | - | Pagination cursor (used by list and `session-events`) |
| `limit` | integer | No | - | Page size (used by list and `session-events`) |
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
{"action": "session", "command": "start", "spec_id": "my-feature-spec-001"}
{"action": "session-step", "command": "next", "session_id": "01HX..."}
{"action": "session-events", "session_id": "01HX...", "limit": 25}
{"action": "complete", "spec_id": "my-spec", "task_id": "task-1-2", "completion_note": "Done"}
{"action": "move", "spec_id": "my-spec", "task_id": "task-1-3", "parent": "phase-2", "position": 1}
{"action": "session-start", "spec_id": "my-feature-spec-001"}
```

`loop_signal` response examples:

```json
{
  "success": true,
  "data": {
    "status": "paused",
    "pause_reason": "phase_complete",
    "loop_signal": "phase_complete"
  },
  "error": null,
  "meta": {"version": "response-v2"}
}
{
  "success": true,
  "data": {
    "status": "paused",
    "pause_reason": "gate_failed",
    "loop_signal": "paused_needs_attention",
    "recommended_actions": [{"action": "review_gate_findings", "description": "Review fidelity findings and apply remediation before retry."}]
  },
  "error": null,
  "meta": {"version": "response-v2"}
}
{
  "success": false,
  "data": {
    "error_code": "AUTHORIZATION",
    "loop_signal": "blocked_runtime",
    "recommended_actions": [{"action": "use_authorized_role", "description": "Switch to a role authorized for session-step actions."}]
  },
  "error": "Authorization denied",
  "meta": {"version": "response-v2"}
}
{
  "success": true,
  "data": {
    "status": "completed",
    "loop_signal": "spec_complete"
  },
  "error": null,
  "meta": {"version": "response-v2"}
}
{
  "success": true,
  "data": {
    "status": "failed",
    "loop_signal": "failed",
    "recommended_actions": [{"action": "collect_failure_context", "description": "Inspect failure details and recent session events before retry."}]
  },
  "error": null,
  "meta": {"version": "response-v2"}
}
```

Operator polling guidance:
- Poll `task(action="session", command="status", session_id=...)` every 10-30 seconds for loop health.
- Read `data.last_step_id`, `data.last_step_type`, `data.current_task_id`, `data.active_phase_progress`, and `data.retry_counters` for operator dashboards.
- Poll `task(action="session-events", session_id=..., cursor=..., limit=...)` for incremental timeline updates.
- Use the [Autonomy Supervisor Runbook](guides/autonomy-supervisor-runbook.md) for preflight, escalation, and stop/continue playbooks.

Role verification preflight (recommended before `session-start`):
- Primary call: `task(action="session", command="list", limit=1)`
- Legacy fallback: `task(action="session-list", limit=1)`
- If response returns `AUTHORIZATION` or `FEATURE_DISABLED`, fail fast and surface remediation (do not start a session).

`session-events` payload schema (`data.events[]`):
- `event_id` (string): Stable event identifier (`{session_id}:{journal_index}`).
- `session_id` (string): Session ID the event belongs to.
- `spec_id` (string): Spec ID for the session.
- `timestamp` (string): Journal timestamp (ISO-8601).
- `event_type` (string): Journal entry type.
- `action` (string, optional): Session action from journal metadata.
- `title` (string): Journal title.
- `summary` (string): Journal content summary.
- `author` (string): Journal author.
- `task_id` (string, optional): Related task ID when present.
- `details` (object, optional): Raw journal metadata for machine consumers.

Pagination rules (`meta.pagination`):
- `cursor` is opaque and session-scoped; reuse only with the same `session_id`.
- Sort order is newest-first by `(timestamp, journal_index)` and remains stable between pages.
- Invalid/mismatched cursors return `error_code=INVALID_CURSOR`.

Proof and receipt requirements (`session-step-next` / `session-step-report`):
- Step reports for issued steps require the one-time `last_step_result.step_proof` token from `data.next_step.step_proof`.
- Proof token semantics are deterministic:
  - same proof + same payload (within replay grace) returns cached response;
  - same proof + different payload returns `STEP_PROOF_CONFLICT`;
  - same proof after replay grace returns `STEP_PROOF_EXPIRED`.
- `execute_verification` with `outcome="success"` requires `last_step_result.verification_receipt`.

Verification receipt construction contract:
- Required fields:
  - `command_hash` (64-char lowercase SHA-256 hex)
  - `exit_code` (integer)
  - `output_digest` (64-char lowercase SHA-256 hex)
  - `issued_at` (timezone-aware ISO-8601 timestamp)
  - `step_id` (must match reported step)
- Binding checks:
  - `verification_receipt.step_id` must match `last_step_result.step_id`
  - `verification_receipt.command_hash` must match the server-issued verification command hash
  - `issued_at` must be within the server-issued verification window for that step

Integrity failure semantics:
- Gate evidence checksum failures return `GATE_INTEGRITY_CHECKSUM`.
- Gate audit mismatches return `GATE_AUDIT_FAILURE`.
- Proof/receipt integrity errors include actionable remediation in `data.details.remediation`.

Scope note:
- Signed/cryptographic verification receipts are deferred; current contract enforces strict field and binding validation on unsigned receipts.

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
| `phase-add-bulk` | Add phase with tasks (auto-appends verification scaffolding) |
| `phase-remove` | Remove a phase |
| `phase-move` | Move phase position |
| `phase-update-metadata` | Update phase metadata |
| `assumption-add` | Add assumption |
| `assumption-list` | List assumptions |
| `revision-add` | Add revision note |
| `constraint-add` | Add constraint |
| `constraint-list` | List constraints |
| `risk-add` | Add risk (with description, likelihood, impact, mitigation) |
| `risk-list` | List risks |
| `question-add` | Add open question |
| `question-list` | List open questions |
| `success-criterion-add` | Add success criterion |
| `success-criteria-list` | List success criteria |
| `spec-find-replace` | Find/replace in spec |
| `spec-rollback` | Rollback to backup |

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | Yes | - | Authoring action |
| `spec_id` | string | Varies | - | Target spec ID |
| `name` | string | Varies | - | Spec name (for spec-create) |
| `template` | string | No | - | Template name |
| `category` | string | No | - | Default task category |
| `plan_path` | string | Yes (spec-create) | - | Path to markdown plan file (relative to specs dir) |
| `plan_review_path` | string | Yes (spec-create) | - | Path to synthesized plan review file (relative to specs dir) |
| `key` | string | Varies | - | Frontmatter key |
| `value` | string | Varies | - | Frontmatter value |
| `phase_id` | string | Varies | - | Phase identifier |
| `position` | integer | No | - | Target position (1-based) |
| `find` | string | No | - | Text/regex to find |
| `replace` | string | No | - | Replacement text |
| `scope` | string | No | `all` | Find-replace scope (`all`, `titles`, `descriptions`) |
| `use_regex` | boolean | No | `false` | Treat find as regex |
| `dry_run` | boolean | No | `false` | Preview changes |
| `description` | string | Varies | - | Risk description (required for `risk-add`) |
| `text` | string | Varies | - | Text content for `constraint-add`, `question-add`, `success-criterion-add` |
| `assumption_type` | string | No | - | Assumption type for `assumption-add` |

### Metadata Actions

The `constraint-add`, `risk-add`, `question-add`, and `success-criterion-add` actions add structured metadata to a spec. Each has a corresponding list action.

**`risk-add` parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `spec_id` | string | Yes | Target spec ID |
| `description` | string | Yes | Risk description |
| `likelihood` | string | No | `low`, `medium`, or `high` |
| `impact` | string | No | `low`, `medium`, or `high` |
| `mitigation` | string | No | Mitigation strategy |

**Plan linkage:** The `spec-create` action requires `plan_path` and `plan_review_path` parameters. These store relative paths (from the specs directory) to the originating markdown plan and its synthesized review. Existing specs without these fields receive validation warnings (not errors). The plan content is included in fidelity review context when available.

**Task complexity:** Tasks support a `metadata.complexity` field with values `low`, `medium`, or `high`. Set via `phase-add-bulk` task metadata. Surfaced in fidelity review context.

### Examples

```json
{"action": "spec-create", "name": "my-new-feature", "plan_path": ".plans/my-new-feature.md", "plan_review_path": ".plan-reviews/my-new-feature-review-full.md"}
{"action": "phase-move", "spec_id": "my-spec", "phase_id": "phase-3", "position": 1}
{"action": "constraint-add", "spec_id": "my-spec", "text": "Must work offline"}
{"action": "risk-add", "spec_id": "my-spec", "description": "API rate limits", "likelihood": "medium", "impact": "high", "mitigation": "Add retry with backoff"}
{"action": "question-add", "spec_id": "my-spec", "text": "Which auth provider?"}
{"action": "success-criterion-add", "spec_id": "my-spec", "text": "All endpoints return <200ms p95"}
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
| `fidelity-gate` | Run autonomous phase gate review and write gate evidence |
| `parse-feedback` | Parse review feedback |
| `list-tools` | List review toolchains |
| `list-plan-tools` | List plan review tools |

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | Yes | - | Review action |
| `spec_id` | string | Varies | - | Target spec ID |
| `phase_id` | string | Varies | - | Phase ID (required for phase-scoped fidelity runs) |
| `session_id` | string | Varies | - | Session ID (required for `fidelity-gate`) |
| `step_id` | string | Varies | - | Session step ID from `task(action=session-step-next)` (required for `fidelity-gate`) |
| `ai_provider` | string | No | - | AI provider to use |
| `ai_timeout` | number | No | 360 | Consultation timeout |
| `consultation_cache` | boolean | No | `true` | Use consultation cache |
| `dry_run` | boolean | No | `false` | Preview without executing |

### Examples

```json
{"action": "spec", "spec_id": "my-spec"}
{"action": "fidelity", "spec_id": "my-spec", "task_id": "task-1-2"}
{"action": "fidelity-gate", "spec_id": "my-spec", "phase_id": "phase-1", "session_id": "session_01", "step_id": "step_01"}
```

**CLI equivalent:** `foundry-cli review spec`

**Plan-enhanced reviews:** When the spec has `metadata.plan_path` pointing to a readable markdown plan, the review automatically enhances to a spec-vs-plan comparison. This evaluates 7 dimensions: coverage, fidelity, success criteria mapping, constraints preserved, risks preserved, open questions preserved, and undocumented additions. The response is structured JSON with a verdict of `aligned`, `deviation`, or `incomplete`. Results are persisted to `specs/.spec-reviews/{spec_id}-spec-review.md` and the path is returned in `review_path`. Specs without `plan_path` continue to receive the standalone full review (backward compatible).

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
{"action": "capabilities"}
```

Capability responses expose both support and runtime-enablement state. Treat
manifest/discovery as hints, and runtime responses as the source of truth for
what is enabled now.

Runtime capability excerpt:

```json
{
  "capabilities": {
    "autonomy_sessions": true,
    "autonomy_fidelity_gates": false
  },
  "runtime": {
    "autonomy": {
      "supported_by_binary": {
        "autonomy_sessions": true,
        "autonomy_fidelity_gates": true
      },
      "enabled_now": {
        "autonomy_sessions": true,
        "autonomy_fidelity_gates": false
      }
    }
  }
}
```

---

## Related

- [CLI Command Reference](04-cli-command-reference.md) - CLI equivalents
- [Response Envelope Guide](concepts/response-envelope.md) - Response format
- [Error Codes Reference](reference/error-codes.md) - Error handling
