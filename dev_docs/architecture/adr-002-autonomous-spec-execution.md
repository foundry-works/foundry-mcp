# ADR-002: Autonomous Spec Execution with Fidelity Gates and Resume

**Status:** Proposed
**Date:** 2026-02-12
**Supersedes:** N/A

## Problem Statement

Foundry MCP can already support parts of autonomous delivery, but it cannot yet run a spec end-to-end in a way that is durable and operationally trustworthy. Current autonomy controls are partial and often ephemeral, so execution can lose continuity across context-window resets, process restarts, or handoffs between callers. This creates state drift risk: what the orchestrator believes happened may diverge from what the coding agent actually did.

Fidelity review also needs to be elevated from a standalone check into a deterministic control loop. The system must define when to advance, when to pause, when to retry, and when to stop, with explicit safeguards for replay/integrity and anti-spinning behavior. It also needs deliberate operating modes (for example, stop at phase boundaries) without discarding progress.

The core problem this ADR solves is execution control, not code generation: provide a durable autonomous session state machine that survives interruption, enforces gate semantics consistently, preserves integrity under concurrency and failure, and supports reliable resume from an exact checkpoint.

## Context

The target workflow is: pass a spec to Foundry MCP, execute implementation autonomously, gate phase progression on fidelity reviews, pause on explicit stop conditions (including context exhaustion and optional phase completion), and resume later without losing state.

Current behavior is close but incomplete:

- `task(action="session-config")` currently supports `get` + `auto_mode` toggles only, with ephemeral in-memory state in `src/foundry_mcp/cli/context.py`.
- Autonomous limits logic exists in `src/foundry_mcp/core/batch_operations.py` (`_check_autonomous_limits`) — the function is exported but has no callers. The guard logic (error threshold, context limit, blocked-task detection) is sound and will be migrated into the new orchestrator module, after which the function and its constants (`MAX_CONSECUTIVE_ERRORS`, `CONTEXT_LIMIT_PERCENTAGE`) will be deleted. Note: the helper `_check_all_blocked()` is also used by `prepare_batch_context` and must be retained or extracted into a shared utility before `_check_autonomous_limits` is removed.
- Fidelity tooling exists (`review(action="fidelity")`) and phase templates already include verify scaffolding (`run-tests`, `fidelity`) in `src/foundry_mcp/core/spec/templates.py`.
- Durable pause/resume architecture already exists in deep research (`src/foundry_mcp/core/research/memory.py`, `src/foundry_mcp/core/research/state_migrations.py`, `src/foundry_mcp/core/research/workflows/deep_research/*`).
- `claude-foundry` skills currently assume richer autonomous session commands (`start/status/pause/resume/end`) than Foundry MCP exposes today.

## Decision

Introduce a durable Autonomous Session Controller (ASC) for spec execution, implemented as two `task` actions — `task(action="session")` for lifecycle management and `task(action="session-step")` for the execution hot path — plus an additive `review` action, with response-v2 envelopes.

Key decisions:

1. Split autonomous session operations into `task(action="session")` for lifecycle commands (`start`, `status`, `pause`, `resume`, `rebase`, `end`, `list`, `reset`) and `task(action="session-step")` for hot-path commands (`next`, `heartbeat`), keeping `session-config` for lightweight config reads during a deprecation window.
2. Persist autonomous session state on disk with schema versioning and migrations (same pattern as deep research).
3. Gate phase progression with explicit fidelity gate records and configurable gate policies (phase cannot advance until gate policy passes).
4. Add context-aware pause/resume using heartbeat updates with staleness detection; keep `status`/`list` read-only and enforce state transitions on mutating paths or maintenance sweeps.
5. Close the orchestration feedback loop: callers must report step outcomes before requesting the next step.
6. Maintain backward compatibility for existing `session-config` (`get`, `auto_mode`) during a deprecation window.
7. Require verifiable gate evidence for `run_fidelity_gate` outcomes using server-generated `gate_attempt_id` matching with session/phase/step binding so callers cannot misattribute or replay gate results. (v1 uses ID matching; cryptographic nonce signing is deferred to v2 for multi-tenant deployments.)
8. Define deterministic active-session lookup behavior (`NO_ACTIVE_SESSION`, `AMBIGUOUS_ACTIVE_SESSION`) when `session_id` is omitted.
9. Enforce one active session per spec — `command="start"` rejects if a non-terminal session already exists for the same `spec_id` (unless `force=true` ends the existing session first).
10. Snapshot spec structure at session start and validate integrity at phase boundaries to detect spec mutations during execution.
11. Make one-active-session-per-spec atomic by guarding `start` with a per-spec lock and active-session index transaction.
12. **[Deferred to v2]** Signed gate evidence (HMAC with key ID rotation) for remote/multi-tenant deployments. v1 uses `gate_attempt_id` matching within the local trust boundary, which is sufficient when caller and server share filesystem trust.
13. Write journal entries for session lifecycle events to provide an audit trail and enrich resume context.
14. Require explicit manual-gate acknowledgment payload on `resume` when paused with `gate_review_required`.
15. Restrict `reset` to `failed` sessions only; reject other source states with deterministic error semantics.
16. Provide a `rebase` command to re-snapshot spec structure after intentional edits, preserving completed-task history when changes are additive. This avoids the `reset` + `start` cycle that discards all session state.
17. Treat heartbeat-reported `context_usage_pct` as cooperative/advisory — it is useful for graceful pausing but is not a hard safety boundary. The true safety boundary is context window exhaustion, which manifests as step staleness when the caller can no longer make API calls.
18. Use ULID format for session and step IDs within the autonomy subsystem. This is scoped to autonomous sessions — it does not change ID formats in other subsystems. If ULIDs prove valuable, adoption in other subsystems is a separate decision.
19. Add configurable stop conditions so the orchestrator pauses when any configured condition is true (logical OR), including phase completion and context exhaustion safeguards.
20. Add a fidelity-cycle stop heuristic (`max_fidelity_review_cycles_per_phase`) to pause sessions that are repeatedly re-running the same phase gate ("spinning wheels").
21. Enable automatic fidelity-gate retries for `strict`/`lenient` policies (configurable) and bound them with the fidelity-cycle safety stop.
22. Prevent gate/step bypass by enforcing autonomy write-locks: while a non-terminal autonomous session exists for a spec, protected task/lifecycle mutations are rejected unless the caller passes a `bypass_autonomy_lock=true` flag. This is a simple boolean guard — the escape hatch is to end/reset the session first. (v1 uses boolean enforcement; cryptographic step-proof tokens with TTL and replay detection are deferred to v2 for multi-tenant deployments.)

Non-goal:

- Foundry MCP will not directly edit code by itself. It provides deterministic orchestration state and gates; the coding agent (for example `claude-foundry` skill) executes implementation steps.

### Design Rationale: Split `session` / `session-step` Actions

We split autonomous session operations into two actions rather than unifying them because:

- **Different performance profiles**: Lifecycle commands (`start`, `status`, `pause`, etc.) have a 5-10s budget and simple validation. The step engine (`next`) has a 15s budget, requires feedback loop validation, gate evidence verification, spec integrity checks, and pause-guard evaluation. Co-locating them would force the internal dispatcher to branch on fundamentally different code paths behind a single entry point.
- **Dispatch depth**: The `task` tool already has 29 actions. A unified `session` action with 10 commands would create three levels of dispatch (tool → action → command) — deeper than any other action in the codebase. Splitting keeps dispatch at two levels, consistent with the existing `ActionRouter` pattern.
- **Independent evolution**: The step engine is the most complex component and the most likely to require iteration. Isolating it in `session-step` allows changes to the hot path without touching lifecycle handler registration or validation.
- **Shared state, separate entry points**: Both actions share session state and apply the same file-locking discipline. Staleness detection runs on both paths. The split is at the dispatch layer, not the state layer — no validation logic is duplicated.

The lifecycle and step actions share common request fields (`session_id`, `workspace`) and the same active-session lookup behavior.

## State Transitions

| From | To | Trigger |
|---|---|---|
| *(none)* | `running` | `command="start"` |
| `running` | `paused` | Auto-pause (context/error/task limit, blocked, gate failure, heartbeat stale, step stale, phase complete stop condition, fidelity-cycle limit) or `command="pause"` |
| `running` | `completed` | `session-step` `command="next"` returns `complete_spec` and all gates passed |
| `running` | `failed` | Unrecoverable error during orchestration (e.g., spec not found, corrupt state after migration failure, spec structure changed) |
| `paused` | `running` | `command="resume"` (only if pause cause is resolved or overridden) |
| `paused` | `running` | `command="rebase"` (re-snapshots spec structure after intentional edits) |
| `failed` | `running` | `command="resume"` with `force=true` (resets error counters, re-validates spec; rejects with `SPEC_REBASE_REQUIRED` if spec structure has changed) |
| `failed` | `running` | `command="rebase"` (re-snapshots spec structure, validates completed tasks exist, resets failure state) |
| `failed` | *(deleted)* | `command="reset"` (hard reset: remove session state + active-session index entry) |
| `running`, `paused`, `failed` | `ended` | `command="end"` |
| `completed`, `ended` | *(terminal)* | No further transitions allowed |

**Passive staleness detection**: Read-path operations (`command="status"`, `command="list"`) do not mutate state. They return `effective_status` plus staleness metadata when heartbeat/step staleness is detected. State transitions to `paused` are applied on mutating paths (`next`, `heartbeat`, `resume`) or by background maintenance sweeps.

## API Contract

### 1) Add `task(action="session")`

Action for autonomous session lifecycle management.

- `command`: `start | status | pause | resume | rebase | end | list | reset`

Common request fields:

- `spec_id` (required for `start`)
- `session_id` (required for `status/pause/resume/rebase/end` unless using active-session lookup; always required for `reset`)
- `workspace` (optional)

Active-session lookup (for `status/pause/resume/rebase/end` when `session_id` omitted):

- If exactly one non-terminal session (`running | paused | failed`) exists in the resolved workspace scope, use it.
- If no non-terminal sessions exist, return `NO_ACTIVE_SESSION`.
- If more than one non-terminal session exists, return `AMBIGUOUS_ACTIVE_SESSION` and require explicit `session_id`.

#### ID Generation

All session and step IDs use ULID format (`auto_01HXYZ...`, `step_01JABCD...`). ULIDs provide monotonic ordering, millisecond precision, and are URL-safe. Generated via the `python-ulid` library (added as a dependency in Phase A). ULID adoption is scoped to the autonomy subsystem — other subsystems retain their existing ID formats.

#### `start` fields

- `idempotency_key` (optional, string, max 128 chars) — Client-generated unique key for this start request. If a session already exists with the same `idempotency_key` and `spec_id`, the server returns the existing session instead of `SPEC_SESSION_EXISTS`. This prevents lost-response retries from failing. Keys are stored in the session state and expire with the session.
- `max_tasks_per_session` (optional, default from config)
- `max_consecutive_errors` (optional, default 3)
- `context_threshold_pct` (optional, default from workflow config)
- `gate_policy` (optional, default `"strict"` — see [Gate Policies](#gate-policies))
- `heartbeat_stale_minutes` (optional, default 10)
- `heartbeat_grace_minutes` (optional, default 5; initial grace window before first heartbeat is required — shorter than `heartbeat_stale_minutes` because the first heartbeat should arrive promptly after the caller begins work)
- `step_stale_minutes` (optional, default 60; max age of an outstanding step before auto-pause)
- `stop_on_phase_completion` (optional, bool, default false — when true, pause after a phase gate passes instead of immediately issuing work from the next phase)
- `max_fidelity_review_cycles_per_phase` (optional, integer, default 3, min 1 — max accepted `run_fidelity_gate` cycles for the active phase before pausing with `fidelity_cycle_limit`)
- `auto_retry_fidelity_gate` (optional, bool, default true — for `strict`/`lenient`, failed gate evaluations automatically schedule remediation (`address_fidelity_feedback`) before retrying the gate, until the phase cycle cap is reached)
- `enforce_autonomy_write_lock` (optional, bool, default true — when true, protected task/lifecycle status mutations are rejected while a non-terminal autonomous session exists, unless `bypass_autonomy_lock=true` is passed)
- `force` (optional, bool, default false — if true and an active session exists for this spec, ends the existing session before starting a new one)

Stop-condition semantics:

- The orchestrator pauses when **any** stop condition is true (logical OR), including context exhaustion safeguards (`context_limit` or `step_stale`), `stop_on_phase_completion`, and the fidelity-cycle cap (`max_fidelity_review_cycles_per_phase`).
- To run "one phase at a time", set `stop_on_phase_completion=true`; execution then stops at the next phase boundary or earlier if context safeguards trigger.

On `start`, the orchestrator:

1. Acquires a per-`spec_id` lock (with a lock-acquisition timeout of 5 seconds; if the lock cannot be acquired, return `LOCK_TIMEOUT` without mutating state). Checks the active-session index for an existing non-terminal session for the same `spec_id`. If found: check `idempotency_key` match first (return existing session if match), then return `SPEC_SESSION_EXISTS` if `force=false`, or end the existing session if `force=true`.
2. Reads the spec and computes a `spec_structure_hash` (SHA-256 of the sorted phase IDs, task IDs, and their parent relationships). Caches the spec file's `mtime` and file size alongside the hash in session state for cheap change detection on subsequent calls.
3. Creates and persists the session state, then atomically updates the active-session index before releasing the lock.
4. Runs opportunistic GC **after** releasing the lock — checks for expired sessions in the same workspace and cleans them (including their index entries and orphaned lock files). GC runs outside the critical section so it does not consume the lock-acquisition budget.
5. Writes a journal entry: `journal(action="add", spec_id=..., entry_type="session", title="Autonomous session started", content="Session {session_id} started with gate_policy={gate_policy}")`.

#### `resume` fields

- `force` (optional, bool, default false — required to resume from `failed`)
- `acknowledge_gate_review` (optional, bool; required and must be `true` when resuming from `pause_reason="gate_review_required"`)
- `acknowledged_gate_attempt_id` (optional string; required when `acknowledge_gate_review=true`, must match the current pending manual gate acknowledgment record)

On `resume`, the orchestrator:

1. If resuming from `failed` with `force=true`: re-reads the spec and re-validates. If `failure_reason` was `spec_structure_changed`, the orchestrator recomputes the spec structure hash. If the structure has actually changed (not just a transient detection), return `SPEC_REBASE_REQUIRED` — the caller must use `rebase` to explicitly accept spec changes. For other failure reasons (`state_corrupt`, `migration_failed`), the orchestrator re-snapshots the spec structure as part of recovery.
2. Writes a journal entry: `journal(action="add", spec_id=..., entry_type="session", title="Session resumed", content="Session {session_id} resumed from {pause_reason}")`.

#### `rebase` fields

- `session_id` (required, or active-session lookup)
- `force` (optional, bool, default false — if true, allows rebase even when completed tasks have been removed from the spec)

`rebase` is valid from `paused` or `failed` states. If status is neither, return `INVALID_STATE_TRANSITION`.

On `rebase`, the orchestrator:

1. Re-reads the spec and computes a new `spec_structure_hash`.
2. If the hash is unchanged, returns success with `rebase_result="no_change"` and transitions to `running` (clears pause/failure state).
3. If the hash differs, computes a structural diff: added phases, removed phases, added tasks, removed tasks.
4. Validates that no completed tasks (tasks in `completed_task_ids`) have been removed from the spec. If completed tasks were removed and `force=false`, return `REBASE_COMPLETED_TASKS_REMOVED` with the list of removed task IDs. If `force=true`, the orchestrator removes those task IDs from `completed_task_ids` and decrements counters accordingly.
5. Updates `spec_structure_hash`, `spec_file_mtime`, and `spec_file_size` in session state.
6. Clears `failure_reason` and `pause_reason`. Transitions to `running`.
7. Returns a `rebase_result` object with the structural diff (added/removed phases and tasks) so the caller knows what changed.
8. Writes a journal entry: `journal(action="add", spec_id=..., entry_type="session", title="Session rebased", content="Session {session_id} rebased. Added: {n} tasks, {m} phases. Removed: {x} tasks, {y} phases.")`.

#### `list` fields

- `status_filter` (optional, e.g. `"running"`, `"paused"`)
- `spec_id` (optional, filter by spec)
- `limit` (optional, default 20, max 100)
- `cursor` (optional, for pagination)

`list` response contract:

- `data.sessions` (array of session summaries)
- `meta.pagination.cursor` (opaque cursor for next page, `null` when exhausted)
- `meta.pagination.has_more` (bool)
- `meta.pagination.page_size` (effective page size)
- `meta.pagination.total_count` (optional; include when inexpensive)
- Cursor ordering is deterministic: `updated_at DESC, session_id DESC`.
- Cursor format is opaque, versioned, and server-generated. Callers must not parse or construct cursors.
- Invalid/malformed cursor returns `INVALID_CURSOR`.

`list` remains read-only. For stale `running` sessions it returns `effective_status="paused"` with staleness metadata (`stale_reason`, `stale_detected_at`) but does not persist the transition.

#### `reset` fields

- `session_id` (required)
- `reason` (optional, string, max 500 chars — recorded in structured logs and the reset metric for post-mortem analysis)

`reset` always requires explicit `session_id` (no active-session lookup). It deletes a `failed` session's state file, removes the session's entry from the active-session index, and returns confirmation. This is the escape hatch when state is corrupt beyond what `force` resume or `rebase` can repair. The caller can then `start` a fresh session for the same spec. The orchestrator writes a journal entry: `journal(action="add", spec_id=..., entry_type="session", title="Session reset", content="Session {session_id} reset. Reason: {reason or 'not provided'}")`.

`reset` is only valid when the current session status is `failed`. If status is not `failed`, return `INVALID_STATE_TRANSITION` and do not mutate state.

#### Response (`response-v2`) data for `start`/`status`/`resume`/`rebase`

```json
{
  "session_id": "auto_01HXYZ...",
  "spec_id": "feature-auth-001",
  "status": "running",
  "pause_reason": null,
  "counters": {
    "tasks_completed": 6,
    "tasks_remaining": 4,
    "consecutive_errors": 0
  },
  "limits": {
    "context_threshold_pct": 85,
    "max_consecutive_errors": 3,
    "max_tasks_per_session": 10,
    "heartbeat_stale_minutes": 10,
    "heartbeat_grace_minutes": 5,
    "step_stale_minutes": 60,
    "max_fidelity_review_cycles_per_phase": 3
  },
  "stop_conditions": {
    "stop_on_phase_completion": false,
    "auto_retry_fidelity_gate": true
  },
  "write_lock_enforced": true,
  "active_phase_id": "phase-2",
  "last_heartbeat_at": "2026-02-13T18:39:12Z",
  "next_action_hint": "task(action='session-step', command='next')"
}
```

**Resume context**: When `command="resume"` or `command="rebase"`, the response includes an additional `resume_context` object to help the caller reconstruct working state after a context window reset:

```json
{
  "resume_context": {
    "spec_id": "feature-auth-001",
    "spec_title": "User Authentication Feature",
    "active_phase_id": "phase-2",
    "active_phase_title": "API endpoint implementation",
    "completed_task_count": 12,
    "recent_completed_tasks": [
      {
        "task_id": "task-2-1",
        "title": "POST /auth/login endpoint",
        "phase_id": "phase-2",
        "files_touched": ["src/routes/auth.py", "tests/test_auth.py"]
      }
    ],
    "completed_phases": [
      {"phase_id": "phase-1", "title": "Data layer", "gate_status": "passed"}
    ],
    "pending_tasks_in_phase": [
      {"task_id": "task-2-2", "title": "POST /auth/register endpoint"},
      {"task_id": "task-2-3", "title": "Token refresh endpoint"}
    ],
    "last_pause_reason": "context_limit",
    "journal_available": true,
    "journal_hint": "Use journal(action='list', spec_id='feature-auth-001') for full implementation history"
  }
}
```

`recent_completed_tasks` is capped at the 10 most recently completed tasks. `completed_task_count` provides the full count. Each entry includes `files_touched` sourced from the corresponding journal entry (if the journal write succeeded for that step; otherwise the field is omitted for that entry). `journal_available` indicates whether journal entries exist for this session — when `false`, the caller should not expect `files_touched` data or follow `journal_hint`. The caller can follow `journal_hint` or re-read specific tasks for deeper context.

### 2) Add `task(action="session-step")`

Action for autonomous session execution hot path: step advancement and heartbeat reporting.

- `command`: `next | heartbeat`

Common request fields:

- `session_id` (required unless using active-session lookup)
- `workspace` (optional)

Active-session lookup follows the same rules as `task(action="session")`: single non-terminal session in workspace scope is used; zero returns `NO_ACTIVE_SESSION`; multiple returns `AMBIGUOUS_ACTIVE_SESSION`.

#### `heartbeat` fields

- `context_usage_pct` (0-100, required)
- `estimated_tokens_used` (optional)
- `consecutive_error_delta` (optional, signed integer)
- `last_completed_task_id` (optional)

**Advisory nature**: `context_usage_pct` is caller-reported and cooperative. The orchestrator uses it for graceful pausing (triggering `context_limit` pause before the caller exhausts its context window), but it is not a hard safety boundary — the orchestrator cannot independently verify context usage. The true safety boundary is context window exhaustion itself, which manifests as step staleness when the caller can no longer make API calls.

#### `next` fields

- `last_step_result` (required after the first call — see [Closing the feedback loop](#closing-the-feedback-loop))

#### `next` response data

- `session_id`
- `status`: `running | paused | completed | failed`
- `state_version` (monotonic write counter incremented on every state mutation; exposed for caller observability — callers can detect when state has changed between reads, but it is not used as an optimistic lock since file locks provide mutual exclusion)
- `next_step` object (or `null` when terminal/paused)

`next_step` always includes a stable `step_id` token. The caller MUST echo this as `last_step_result.step_id` on the next `command="next"` call.

`next_step.type` enum:

- `implement_task`
- `execute_verification`
- `run_fidelity_gate`
- `address_fidelity_feedback`
- `pause`
- `complete_spec`

`next_step.instructions` is an array of structured hint objects describing the recommended tool calls for the step. These are guidance for the caller, not executable commands — the caller is responsible for translating them into actual API calls with appropriate parameters.

`address_fidelity_feedback` is a required remediation step in auto-retry flows. The caller should apply fixes from the latest gate findings, then report the remediation outcome via `last_step_result` before the orchestrator issues the next gate retry.

`next_step` examples:

```json
{
  "step_id": "step_01JABCD...",
  "type": "implement_task",
  "task_id": "task-2-3",
  "phase_id": "phase-2",
  "task_title": "Token refresh endpoint",
  "instructions": [
    {"tool": "task", "action": "prepare", "description": "Load task context and acceptance criteria"},
    {"tool": "task", "action": "start", "description": "Mark task as in-progress"},
    {"tool": "task", "action": "complete", "description": "Mark task as complete after implementation"}
  ]
}
```

```json
{
  "step_id": "step_01JABCE...",
  "type": "run_fidelity_gate",
  "phase_id": "phase-2",
  "instructions": [
    {"tool": "review", "action": "fidelity-gate", "description": "Run phase fidelity review against spec requirements"}
  ]
}
```

```json
{
  "step_id": "step_01JABCG...",
  "type": "address_fidelity_feedback",
  "phase_id": "phase-2",
  "gate_attempt_id": "gate_01J...",
  "instructions": [
    {"tool": "journal", "action": "list", "description": "Read latest gate findings and remediation recommendations"},
    {"tool": "task", "action": "update", "description": "Apply code/test/doc changes needed to satisfy failed gate findings"}
  ]
}
```

```json
{
  "step_id": "step_01JABCF...",
  "type": "pause",
  "reason": "context_limit",
  "message": "Context usage at 87% (threshold: 85%). Resume in a new session."
}
```

### 3) Backward Compatibility for `session-config`

`task(action="session-config")` retains its existing `get` field unchanged. During the deprecation window:

- `auto_mode=true` logs a deprecation warning and delegates to `session` `command="start"`. Requires `spec_id` as an explicit parameter. If `spec_id` is omitted, return `AUTO_MODE_SPEC_REQUIRED` with a message directing the caller to use `task(action="session", command="start", spec_id=...)` instead. No implicit spec resolution — the heuristic of scanning the workspace for a single active spec is fragile for a deprecation path and not worth the implementation cost.
- `auto_mode=false` logs a deprecation warning and delegates to `session` `command="pause"` via active-session lookup in the resolved workspace. If no active session exists, return `NO_ACTIVE_SESSION`.
- `get=true` continues to return the lightweight session config (not the full session state).

After two release cycles (Phase F), `auto_mode` support is removed.

### 4) Closing the feedback loop

After the caller executes the step returned by a previous `session-step` `command="next"` call, it **must** report the outcome via `last_step_result` on the next call:

```json
{
  "last_step_result": {
    "step_id": "step_01JABCD...",
    "step_type": "implement_task",
    "task_id": "task-2-3",
    "outcome": "success",
    "note": "Implemented and tests passing",
    "files_touched": ["src/routes/auth.py", "tests/test_auth.py"]
  }
}
```

`outcome` enum: `success | failure | skipped`

- `success`: Step completed. Orchestrator advances state (marks task complete, records verification, etc.).
- `failure`: Step execution failed. Orchestrator increments `consecutive_errors`. On `implement_task` failure, the task remains in its current status for retry.
- `skipped`: Caller chose to skip (e.g., manual override). Orchestrator logs a warning and moves on.

`files_touched` (optional, array of strings): File paths modified during this step. The orchestrator writes these into a journal entry for the step (not into session state). This data is caller-reported and advisory — it is useful for resume context and audit but is not authoritative. The orchestrator does not execute code and cannot verify these paths.

`note` (optional, string): Free-text note about the step outcome. Included in the journal entry.

For `step_type="run_fidelity_gate"`:

- `last_step_result` MUST include `phase_id` and `gate_attempt_id`.
- `review(action="fidelity-gate")` creates a gate-attempt record in session state (`pending_gate_evidence`) scoped to `{session_id, phase_id, step_id, gate_attempt_id, verdict, issued_at}`.
- Multiple review attempts are allowed for the same `step_id` before `command="next"` consumes one. Each new attempt replaces `pending_gate_evidence` for that step (`latest attempt wins`).
- `command="next"` validates `gate_attempt_id` against `pending_gate_evidence`: matching attempt ID, matching session/phase/step binding. If invalid, return `INVALID_GATE_EVIDENCE` without mutating state.
- After validation, orchestrator derives gate pass/fail from `verdict` + session `gate_policy` (caller-provided pass/fail booleans are non-authoritative).
- Each accepted `run_fidelity_gate` result increments the active phase's fidelity-review cycle counter. The counter resets when the session advances to a new phase.
- Gate failure handling:
  - `gate_policy="manual"` always pauses with `gate_review_required`.
  - `gate_policy in {"strict","lenient"}` with `auto_retry_fidelity_gate=true` schedules `address_fidelity_feedback` while below `max_fidelity_review_cycles_per_phase`.
  - After `address_fidelity_feedback` is reported as `outcome="success"`, the orchestrator schedules the next `run_fidelity_gate` retry cycle.
  - Otherwise pause with `gate_failed`.
  - Gate failures do **not** increment `consecutive_errors`.

For `step_type="address_fidelity_feedback"`:

- `last_step_result.note` should summarize how gate findings were addressed.
- `last_step_result.files_touched` should include remediation edits when available.
- On `outcome="success"`, the orchestrator schedules `run_fidelity_gate` for the retry cycle (subject to cycle limits).

**v1 trust model**: Gate evidence uses `gate_attempt_id` matching within the local trust boundary. The `gate_attempt_id` is server-generated and bound to `{session_id, phase_id, step_id}`, preventing misattribution. Since v1 callers share filesystem trust with the server, this is sufficient. Cryptographic nonce signing (HMAC with key ID rotation) for remote/multi-tenant deployments is deferred to v2.

If `last_step_result` is omitted on a non-initial call, the orchestrator returns an error (`STEP_RESULT_REQUIRED`) rather than silently advancing. The first `command="next"` call after `start` or `resume`/`rebase` is exempt.

### 5) Add `review(action="fidelity-gate")`

Purpose: run a phase fidelity review and return the gate result. The review tool writes pending gate-attempt evidence metadata only (no task/session progression), and the orchestrator (`session-step` `command="next"` via `last_step_result`) remains the sole owner of gate outcome transitions and phase/session state changes.

Request fields:

- `spec_id` (required)
- `session_id` (required)
- `phase_id` (required)
- `step_id` (required — the step ID from the `run_fidelity_gate` next_step)
- Existing fidelity inputs (`ai_tools`, `model`, `consensus_threshold`, etc.)

Response data:

- `spec_id`
- `session_id`
- `phase_id`
- `step_id`
- `gate_attempt_id` (unique per review attempt for the given `step_id`)
- `verdict`: `pass | fail | warn`
- `gate_policy` (echo of active session policy for transparency)
- `gate_passed_preview` (bool, convenience preview only; `command="next"` recomputes authoritatively)
- `review_path` (if written)
- `findings` (summary of issues, if any)

If review execution succeeds but gate fails, return `success=true` with warnings and `gate_passed_preview=false`. The caller still reports the step as `outcome="success"` with `gate_attempt_id`; `session-step` `command="next"` validates the attempt ID against the pending gate record and applies gate policy. If policy fails, orchestrator pauses the session without incrementing `consecutive_errors`.

Retry semantics for repeated reviews on the same gate step:

- Re-running `review(action="fidelity-gate")` with the same `{session_id, phase_id, step_id}` creates a new `gate_attempt_id` and replaces the prior pending attempt record.
- `command="next"` accepts only the latest pending attempt for that step.
- Once `command="next"` consumes a valid attempt and records the gate outcome, additional submissions for that step are rejected via normal step progression checks (`STEP_MISMATCH`), because `last_step_issued` has advanced.

### 6) Non-Bypass Enforcement

To prevent callers from bypassing gate/step sequencing by directly mutating task or phase state, autonomous sessions enforce a write-lock when `enforce_autonomy_write_lock=true`.

Protected actions (non-exhaustive):

- `task` mutations that change execution/progress state (`start`, `complete`, `update-status`, `block`, `unblock`, dependency/status mutations)
- lifecycle/status mutations that can effectively skip orchestration sequencing

Rules:

- While a non-terminal autonomous session (`running | paused | failed`) exists for a spec and write-lock is enabled, protected mutations are rejected with `AUTONOMY_WRITE_LOCK_ACTIVE`.
- Protected mutation requests can pass `bypass_autonomy_lock=true` to override — this is logged as a warning and recorded in the journal. The bypass flag is an explicit opt-out for manual intervention, not a routine code path.
- Read-only actions remain unaffected.
- Controlled escape hatch: end/reset the autonomous session first (`session` lifecycle commands), then perform manual status surgery explicitly outside autonomy mode.

This means an agent cannot accidentally "just mark task/phase complete" through alternate MCP status-mutating actions while the autonomous controller is active. The boolean guard is sufficient for v1 (single local caller). Cryptographic step-proof tokens (with TTL, replay detection, and allowed-action-set binding) are deferred to v2 for multi-tenant deployments where the bypass flag would be an insufficient trust boundary.

### 7) Extend `server(action="capabilities")`

Add capability flags for negotiation/discovery:

- `autonomy_sessions` — core lifecycle (`session`) and step engine (`session-step`) available, including resume, rebase, heartbeat, and write-lock support
- `autonomy_fidelity_gates` — gate enforcement and auto-retry available
- `autonomy_signed_gate_evidence` — **[v2]** remote trust mode with cryptographic nonce signing

Callers check `autonomy_sessions` to know the feature exists, and `autonomy_fidelity_gates` to know gate enforcement is active. All other settings (write-lock enforcement, auto-retry policy, heartbeat config) are session-level configuration communicated in the `start` response — they do not need capability negotiation because callers see them directly in the session state.

Expose rollout state and defaults in `capabilities.feature_flags`.

## Gate Policies

The `gate_policy` field on session start controls how fidelity gate verdicts are evaluated:

| Policy | Behavior | Use case |
|---|---|---|
| `strict` (default) | Gate passes only on `verdict="pass"`. `warn` or `fail` triggers automatic remediation+re-review when `auto_retry_fidelity_gate=true` and cycle cap not reached; otherwise pauses with `gate_failed`. | Production-quality specs, critical paths |
| `lenient` | Gate passes on `pass` or `warn`. `fail` triggers automatic remediation+re-review when `auto_retry_fidelity_gate=true` and cycle cap not reached; otherwise pauses with `gate_failed`. | Iterative development, early-phase work |
| `manual` | Gate never auto-passes. Always pauses with `gate_review_required`, presenting the verdict for human decision. The caller must `resume` with explicit acknowledgment fields (`acknowledge_gate_review=true` + `acknowledged_gate_attempt_id`). | High-risk phases, compliance-sensitive work |

Gate policy is set at session start and applies uniformly to all phases. Per-phase policy overrides are a future extension (tracked, not implemented in v1).

Automatic retry behavior:

- `auto_retry_fidelity_gate=true` (default) applies only to `strict` and `lenient`.
- Failed strict/lenient gates first emit `address_fidelity_feedback`; only after that step reports success does the orchestrator issue the next `run_fidelity_gate` retry.
- Retries are bounded by `max_fidelity_review_cycles_per_phase`; when the cap is reached, the session pauses with `fidelity_cycle_limit`.
- `manual` policy ignores auto-retry and always requires explicit human acknowledgment on resume.

Manual-gate resume rules:

- When a session is paused with `pause_reason="gate_review_required"`, `command="resume"` requires `acknowledge_gate_review=true` and `acknowledged_gate_attempt_id`.
- `acknowledged_gate_attempt_id` must match the current `pending_manual_gate_ack.gate_attempt_id` in session state.
- Missing acknowledgment returns `MANUAL_GATE_ACK_REQUIRED`; mismatched or stale attempt ID returns `INVALID_GATE_ACK`.

## Spec Integrity Validation

At session start, the orchestrator computes a `spec_structure_hash` — a SHA-256 digest of the spec's sorted phase IDs, task IDs, task-to-phase parent mappings, and phase ordering. This hash and the spec file's `mtime` + file size are stored in session state.

On each `session-step` `command="next"` call, the orchestrator uses a two-tier change detection strategy:

1. **Fast path (metadata check)**: Compare current `mtime` and file size against cached values. If both are unchanged and the current call is not crossing a phase boundary, skip re-hashing and proceed.
2. **Slow path (re-hash)**: If either metadata value changed, or if a phase boundary is being crossed, re-read the spec structure and recompute the hash.

Hash comparison outcomes:

- **Match** (or metadata unchanged): Proceed normally. Update cached `mtime` and file size if the file was re-read.
- **Mismatch**: Transition to `failed` with `failure_reason="spec_structure_changed"`. The response includes a `spec_drift` object describing what changed (added/removed phases or tasks).

This prevents silent state corruption when the spec is edited mid-session. When spec drift is detected, the caller has two recovery paths:

1. **`rebase`** (preferred): Use `session` `command="rebase"` to explicitly accept the spec changes. The orchestrator validates that completed tasks still exist in the new spec structure (unless `force=true`), re-snapshots the hash, and returns to `running`. This preserves all session state — completed tasks, gate records, and counters.
2. **`reset` + `start`** (escape hatch): Use `session` `command="reset"` to delete the session entirely, then `start` a fresh session. This discards all session state but preserves task completion statuses that live in the spec itself.

**Scope**: Only structural changes (added/removed/reordered phases and tasks) trigger failure. Changes to task descriptions, acceptance criteria, or other metadata are tolerated — these don't affect the orchestrator's step sequencing.

## Journal Integration

The orchestrator writes journal entries for key session lifecycle events using `journal(action="add")`. This provides an audit trail, enriches resume context, and makes session history queryable via the existing journal API.

Events that produce journal entries:

| Event | `entry_type` | Content includes |
|---|---|---|
| Session started | `session` | `session_id`, `gate_policy`, `spec_id` |
| Session paused | `session` | `session_id`, `pause_reason` |
| Session resumed | `session` | `session_id`, previous `pause_reason` |
| Session rebased | `session` | `session_id`, structural diff summary |
| Session completed | `session` | `session_id`, `tasks_completed`, total duration |
| Session ended | `session` | `session_id`, reason (user-initiated) |
| Session reset | `session` | `session_id`, `reason` (caller-provided) |
| Task step completed | `step` | `step_id`, `task_id`, `outcome`, `files_touched`, `note` |
| Gate evaluated | `gate` | `phase_id`, `verdict`, `gate_passed`, review findings summary |

Journal writes are best-effort for mid-session events — a journal write failure does not block session state transitions. Failures are logged with `task.session.journal_write_failed.total` metric.

**Exception**: The `start` journal write is mandatory. If the journal write fails during `command="start"`, the orchestrator rolls back the session (deletes state file and index entry) and returns an error. The rationale: a session that starts without an audit trail has no provenance from inception, which undermines the value of journal-based resume context. Since `start` is an infrequent operation and the journal write is a cheap local I/O, the risk of spurious failures is low.

The `resume_context` includes `files_touched` inline for recent completed tasks (sourced from journal entries), and `journal_available` indicates whether journal writes succeeded. When `journal_available` is `true`, `journal_hint` directs callers to the journal for full history. When `false`, the inline `resume_context` data is the best available context.

## Autonomous Session State Schema

Persist as JSON (file-backed), versioned independently from response envelope. `_schema_version` starts at 1 — there are no pre-existing v0 state files since autonomous sessions are a new subsystem. The migration infrastructure is included from the start (mirroring the deep-research pattern) so that future schema changes can be applied without manual intervention.

```json
{
  "_schema_version": 1,
  "id": "auto_01HXYZ...",
  "spec_id": "feature-auth-001",
  "idempotency_key": null,
  "spec_structure_hash": "sha256:e3b0c44298fc...",
  "spec_file_mtime": 1739474040.123,
  "spec_file_size": 18204,
  "status": "running",
  "created_at": "2026-02-13T18:14:00Z",
  "updated_at": "2026-02-13T18:39:12Z",
  "paused_at": null,
  "pause_reason": null,
  "failure_reason": null,
  "active_phase_id": "phase-2",
  "last_task_id": "task-2-2",
  "last_step_issued": {
    "step_id": "step_01JABCD...",
    "type": "implement_task",
    "task_id": "task-2-3",
    "issued_at": "2026-02-13T18:39:12Z"
  },
  "pending_gate_evidence": null,
  "pending_manual_gate_ack": null,
  "counters": {
    "tasks_completed": 6,
    "consecutive_errors": 0,
    "fidelity_review_cycles_in_active_phase": 1
  },
  "limits": {
    "max_tasks_per_session": 10,
    "max_consecutive_errors": 3,
    "context_threshold_pct": 85,
    "heartbeat_stale_minutes": 10,
    "heartbeat_grace_minutes": 5,
    "step_stale_minutes": 60,
    "max_fidelity_review_cycles_per_phase": 3
  },
  "stop_conditions": {
    "stop_on_phase_completion": false,
    "auto_retry_fidelity_gate": true
  },
  "write_lock_enforced": true,
  "gate_policy": "strict",
  "context": {
    "context_usage_pct": 72,
    "estimated_tokens_used": 113400,
    "last_heartbeat_at": "2026-02-13T18:39:12Z"
  },
  "phase_gates": {
    "phase-1": {
      "required": true,
      "status": "passed",
      "verdict": "pass",
      "gate_attempt_id": "gate_01J...",
      "review_path": "specs/.fidelity-reviews/feature-auth-001-phase-phase-1.md",
      "evaluated_at": "2026-02-13T18:20:03Z"
    }
  },
  "completed_task_ids": ["task-1-1", "task-1-2", "task-2-1", "task-2-2"],
  "state_version": 7
}
```

Schema fields of note:

- `state_version`: Monotonic write counter incremented on every state mutation. Used for caller observability (detecting state changes between reads), not as an optimistic lock — file locks provide mutual exclusion.
- `idempotency_key`: Client-provided key from `start`, or `null`. Stored for duplicate-start detection. Expires with the session.
- `spec_file_mtime`: Cached mtime of the spec file at last hash computation.
- `spec_file_size`: Cached spec file size used with `spec_file_mtime` for fast-path change detection.
- `pending_gate_evidence`: Current unconsumed gate attempt for the active `run_fidelity_gate` step. Includes `gate_attempt_id`, `step_id`, `phase_id`, `verdict`, and `issued_at`. Replaced on each new review attempt for the same step.
- `pending_manual_gate_ack`: Set when `gate_policy="manual"` pauses the session. Includes the gate attempt the caller must acknowledge on `resume`.
- `max_fidelity_review_cycles_per_phase`: Spin-detection guard for repeated gate attempts in the same phase.
- `stop_conditions.stop_on_phase_completion`: When `true`, the orchestrator pauses with `pause_reason="phase_complete"` after a phase gate passes and before issuing work from the next phase.
- `stop_conditions.auto_retry_fidelity_gate`: When `true`, failed `strict`/`lenient` gate outcomes automatically schedule `address_fidelity_feedback` followed by another `run_fidelity_gate` while below the phase cycle cap.
- `write_lock_enforced`: Enables protected-mutation boolean lockout for non-terminal autonomous sessions. Protected mutations are rejected with `AUTONOMY_WRITE_LOCK_ACTIVE` unless `bypass_autonomy_lock=true` is passed.

Enums:

- `status`: `running | paused | completed | failed | ended`
- `pause_reason` (why the session paused): `user | context_limit | error_threshold | blocked | gate_failed | gate_review_required | task_limit | heartbeat_stale | step_stale | phase_complete | fidelity_cycle_limit`
- `failure_reason` (why the session failed): `spec_not_found | spec_structure_changed | state_corrupt | migration_failed`
- `phase_gates.*.status`: `pending | passed | failed | waived`

## Orchestration Rules

`task(action="session-step", command="next")` follows this order:

1. **Validate feedback**: If not the first call in the session/resume/rebase, require `last_step_result`. Return `STEP_RESULT_REQUIRED` error if missing.
2. **Validate step identity**: If `last_step_issued` doesn't match the reported step (`step_id`/type/task-or-phase binding), return `STEP_MISMATCH`.
3. **Record step outcome**: Apply `last_step_result` to session state (mark task complete, increment error counter, record gate result, etc.). Write a journal entry for the step outcome (including `files_touched` and `note` if provided). For `run_fidelity_gate`, validate `gate_attempt_id` against `pending_gate_evidence` (matching attempt ID, matching session/phase/step binding); on invalid/stale attempt return `INVALID_GATE_EVIDENCE` and do not mutate state. On valid gate consumption, increment `fidelity_review_cycles_in_active_phase`.
4. **Validate spec integrity**: Compare spec file `mtime` + file size to cached values. If either changed, or if this call crosses a phase boundary, recompute `spec_structure_hash` and compare to stored value. On mismatch, transition to `failed` with `failure_reason="spec_structure_changed"`.
5. **Check terminal states**: If session is `completed` or `ended`, return terminal `status` with `next_step=null`.
6. **Enforce step staleness** (hard backstop): If `last_step_issued` is non-null and `now - last_step_issued.issued_at > step_stale_minutes`, pause (`step_stale`). This catches sessions where the caller received a step, partially executed, and disappeared without reporting back.
7. **Enforce heartbeat staleness** (cooperative signal):
   - Before first heartbeat, allow a grace window (`heartbeat_grace_minutes`) from `created_at`.
   - After first heartbeat, if `now - last_heartbeat_at > heartbeat_stale_minutes`:
     - If `last_step_issued` is non-null and the step is not yet stale (i.e., within `step_stale_minutes`), include a `heartbeat_stale_warning=true` flag in `data.details` but do **not** pause. The caller may simply be busy executing the step.
     - If no outstanding step exists (session is idle between steps), pause (`heartbeat_stale`) and return `HEARTBEAT_STALE` in `data.details.pause_trigger`.
   - This ordering ensures the step staleness check (hard backstop) takes priority, and heartbeat staleness (cooperative signal) does not prematurely interrupt active work.
8. **Enforce pause guards (cooperative)**:
   - `context_usage_pct >= context_threshold_pct` -> pause (`context_limit`). Note: this guard relies on caller-reported heartbeat data and is advisory (see decision 17).
   - `consecutive_errors >= max_consecutive_errors` -> pause (`error_threshold`)
   - `tasks_completed >= max_tasks_per_session` -> pause (`task_limit`)
9. **Enforce fidelity-cycle stop heuristic**: If current phase gate is not yet passed and `fidelity_review_cycles_in_active_phase >= max_fidelity_review_cycles_per_phase`, pause (`fidelity_cycle_limit`).
10. If all pending work is blocked -> pause (`blocked`).
11. If current phase implementation tasks are complete and verifications pending -> `execute_verification`.
12. If phase verifications complete and fidelity gate pending -> `run_fidelity_gate`.
13. If latest fidelity verdict fails policy:
   - `gate_policy="manual"` -> pause (`gate_review_required`).
   - `gate_policy in {"strict","lenient"}` and `auto_retry_fidelity_gate=true` and cycle cap not reached -> `address_fidelity_feedback` (agent must apply fixes before the next retry).
   - Otherwise -> pause (`gate_failed`).
14. If `address_fidelity_feedback` was completed successfully and current phase gate remains unpassed -> `run_fidelity_gate` (retry cycle).
15. If the current phase gate has passed and `stop_on_phase_completion=true` and additional phases/tasks remain, pause (`phase_complete`).
16. If gate passed and next unblocked task exists -> `implement_task`.
17. If no remaining tasks -> `complete_spec` then transition session `status=completed`. Write a journal entry for session completion.

On success, set `last_step_issued` to the returned step and persist before responding.

On any pause transition, write a journal entry recording the `pause_reason`.

`task(action="session", command="status")` and `command="list"` are read-only views:

- They compute staleness and expose `effective_status` (`running` or derived `paused`) without mutating persisted state.
- They include `stale_reason` (`heartbeat_stale` or `step_stale`) when derived pause applies.
- Consumers should call `resume` after resolving stale causes; `session-step` `next`/maintenance sweeps apply the persisted pause transition.

`task(action="session", command="resume")` manual-gate rules:

- If `pause_reason` is `gate_review_required`, require `acknowledge_gate_review=true` and `acknowledged_gate_attempt_id`.
- `acknowledged_gate_attempt_id` must match `pending_manual_gate_ack.gate_attempt_id`.
- On success, clear `pending_manual_gate_ack` and transition to `running`.
- On failure, return `MANUAL_GATE_ACK_REQUIRED` or `INVALID_GATE_ACK` without mutating state.

## Timeout and Cancellation Budgets

Default timeout budgets for `task(action="session")`:

| Command(s) | Default | Max override |
|---|---|---|
| `status`, `list`, `pause`, `resume`, `end`, `reset` | 5s | 15s |
| `rebase` | 10s | 30s |
| `start` | 10s | 30s |

Default timeout budgets for `task(action="session-step")`:

| Command(s) | Default | Max override |
|---|---|---|
| `heartbeat` | 5s | 15s |
| `next` | 15s | 45s |

Default timeout budgets for `review`:

| Action | Default | Max override |
|---|---|---|
| `fidelity-gate` | 120s | 300s |

Rules:

- All long-running operations use explicit timeouts; no indefinite waits.
- **Lock acquisition timeout**: Per-spec lock acquisition uses a 5-second timeout (included within the command's overall budget). If the lock cannot be acquired, return `LOCK_TIMEOUT` with `error_type="unavailable"`. No state is mutated.
- **Opportunistic GC runs outside the lock scope** for `start` — session creation and index update happen within the lock; GC of expired sessions happens after lock release. This prevents GC I/O from consuming the lock-acquisition budget of concurrent callers.
- Timeout/cancel before persistence must leave state unchanged.
- Timeout/cancel after persistence returns `TIMEOUT` with `error_type="unavailable"` and `data.details.state_version` so callers can reconcile via `status`.
- `review(action="fidelity-gate")` timeout does not mutate session state; caller retries review or pauses session.

## Persistence, Migration, and Concurrency

- Storage location: `specs/.autonomy/sessions/` (workspace default), fallback `~/.foundry-mcp/autonomy/sessions/`.
- Active-session index: `specs/.autonomy/index/spec_active_index.json` (workspace default), fallback under user home.
- Per-spec locks: `specs/.autonomy/locks/spec_<spec_id>.lock`.
- One JSON file per session, named by session ID.
- Use file locks (`filelock`) and sanitized IDs (same security pattern as research memory).
- **Filesystem requirement**: File locking assumes a local POSIX filesystem. Network filesystems (NFS, CIFS) are not supported for session storage because `fcntl.flock` is advisory-only and does not guarantee mutual exclusion across hosts. If `specs/` is on a network mount, the user should configure the fallback path (`~/.foundry-mcp/autonomy/sessions/`) on local storage.
- Add migration module (`_schema_version`) mirroring deep-research migration strategy.

### Concurrency Model

File locks (via `filelock`) prevent torn reads and writes. Each state-mutating operation acquires an exclusive lock on the session file, reads, mutates, increments `state_version`, and writes atomically within the lock.

One-active-session-per-spec is enforced atomically:

1. `start` acquires `spec_<spec_id>.lock` (5-second lock-acquisition timeout; `LOCK_TIMEOUT` on failure).
2. Reads `spec_active_index.json`.
3. Checks `idempotency_key` match (return existing session if match). Validates existing non-terminal session for `spec_id` (`SPEC_SESSION_EXISTS` unless `force=true`).
4. Creates/updates session file.
5. Commits index update in the same lock scope.
6. Releases lock.
7. Runs opportunistic GC outside the lock scope (see [Timeout and Cancellation Budgets](#timeout-and-cancellation-budgets)).

The remaining risk is a single session being driven by two callers simultaneously (e.g., a user manually calls `next` while a skill is also driving the session). Session file locking makes this safe at the data level. At the protocol level, the second caller's `next` call fails with `STEP_MISMATCH` because `last_step_issued` will differ — this rejects concurrent driving deterministically.

`state_version` is retained in the schema for caller observability (detecting when state has changed between reads) but is not used as an optimistic lock. File locks provide the necessary mutual exclusion.

### Session State Recovery

When session state is corrupt (e.g., unparseable JSON, failed migration):

1. `command="status"` returns `status="failed"` with `failure_reason="state_corrupt"` or `"migration_failed"`.
2. `command="resume"` with `force=true` attempts to recover by re-reading the spec, resetting counters, and re-validating. If the spec structure has changed, returns `SPEC_REBASE_REQUIRED` instead (the caller should use `rebase`). If recovery fails, returns `SESSION_UNRECOVERABLE`.
3. `command="rebase"` re-snapshots the spec structure and clears the failure state. This is the preferred recovery path when the spec was intentionally edited.
4. `command="reset"` (allowed only when status is `failed`) deletes the session state file and removes the session from the active-session index. The caller can then `start` a new session. Task completion state from the spec itself (task statuses) is preserved since those live in the spec, not the session.

### Session Cleanup and Garbage Collection

- Sessions in terminal states (`completed`, `ended`) are eligible for cleanup after a configurable TTL (default: 7 days).
- `failed` sessions are retained for 30 days (longer window for debugging).
- `task(action="session", command="list")` returns all sessions with their status, enabling manual management.
- **Primary GC trigger**: Opportunistic GC runs during `command="start"` calls (after lock release) — the orchestrator checks for expired sessions in the same workspace and cleans them (including their index entries). This ensures GC runs even in short-lived CLI invocations where a background loop would never fire.
- **Lock file cleanup**: GC also removes orphaned lock files in `specs/.autonomy/locks/` whose corresponding sessions no longer exist. Lock files are small but accumulate over time (one per spec that ever had a session).
- **Secondary GC trigger**: An in-process maintenance loop (default every 6 hours) runs GC and stale-session reconciliation when the server is long-running. `status`/`list` remain read-only (they do not persist stale-to-paused transitions).
- Cleanup is logged with `task.session.gc.total{status=...}` metric.

## Observability and Security

Observability:

- Metrics:
  - `task.session.start.total`
  - `task.session.pause.total{reason=...}`
  - `task.session.resume.total`
  - `task.session.rebase.total{result=no_change|success|completed_tasks_removed}`
  - `task.session.reset.total{reason=provided|not_provided}`
  - `task.session.gc.total{status=...}`
  - `task.session_step.next.duration_ms`
  - `task.session_step.next.step_result{outcome=success|failure|skipped}`
  - `task.session_step.heartbeat.stale.total`
  - `task.session_step.step.stale.total`
  - `task.session_step.gate_retry.total`
  - `task.autonomy_write_lock.reject.total`
  - `task.autonomy_write_lock.bypass.total`
  - `task.session.stale.detected.total{source=status|list|next|maintenance}`
  - `task.session.spec_drift.total`
  - `task.session.timeout.total{command=...}`
  - `task.session_step.timeout.total{command=...}`
  - `task.session.lock_timeout.total{command=...}`
  - `task.session.journal_write_failed.total`
  - `review.fidelity_gate.timeout.total`
  - `review.fidelity_gate.total{outcome=pass|fail|warn}`
- Structured logs with `request_id`, `session_id`, `spec_id`, `phase_id`, `step_id`, `pause_reason`, `failure_reason`, `state_version`.

Security:

- Validate all IDs and bounds (`context_usage_pct` 0-100, thresholds positive integers, `max_fidelity_review_cycles_per_phase >= 1`, `auto_retry_fidelity_gate` boolean, `enforce_autonomy_write_lock` boolean, counters non-negative).
- Sanitize text fields (task titles, notes, pause reasons, reset reasons, rebase reasons) before logs/responses.
- Enforce workspace path constraints and existing trust boundary checks.
- Treat caller-reported step outcomes as untrusted until matched against `last_step_issued` and (for gates) validated via `gate_attempt_id` binding.
- Treat caller-reported `context_usage_pct` as advisory — use for cooperative pausing but do not rely on it as a security boundary.
- Validate `gate_attempt_id` against `pending_gate_evidence` (matching session/phase/step binding) before accepting gate results.
- Validate manual-gate acknowledgment payloads (`acknowledge_gate_review`, `acknowledged_gate_attempt_id`) against `pending_manual_gate_ack` before allowing resume.
- Enforce autonomy write-lock (boolean check) on protected mutations when a non-terminal session exists and `enforce_autonomy_write_lock=true`. Log bypass usage.
- No shell execution introduced by session APIs.
- `force` resume from `failed` requires explicit flag to prevent accidental resumption of broken sessions.
- `force` rebase requires explicit flag to prevent accidental loss of completed-task records.
- `idempotency_key` is validated for length (max 128 chars) and character set (alphanumeric, hyphens, underscores) to prevent injection.
- **[v2]** Signed gate evidence tokens (HMAC with key rotation) and cryptographic step-proof tokens for remote/multi-tenant deployments where filesystem trust is not shared.

## Rollout Plan

1. Phase A (feature-flagged): persistent session store + `task(action="session")` lifecycle commands (`start`, `status`, `pause`, `resume`, `end`, `list`, `reset`). Add `python-ulid` dependency. Register `task(action="session-step")` action with feature-flag guard.
2. Phase B: `session-step` `command="next"` step engine with feedback loop, pause guards, heartbeat staleness, step staleness, phase-complete stop condition, fidelity-cycle stop heuristic, boolean autonomy write-lock enforcement for protected mutations, spec integrity validation, and journal integration.
3. Phase C: `review(action="fidelity-gate")` and gate policy enforcement.
4. Phase D: `session` `command="rebase"` with structural diff computation and completed-task validation.
5. Phase E: `claude-foundry` skill + hook alignment (`session` lifecycle + `session-step` hot path).
6. Phase F: deprecate legacy `session-config` `auto_mode` path after two release cycles.
7. **[v2]** Cryptographic step-proof tokens, signed gate evidence (HMAC), and `autonomy_signed_gate_evidence` capability flag for multi-tenant deployments.

Feature flags (initial defaults):

- `autonomy_sessions`: off
- `autonomy_fidelity_gates`: off

## Legacy Code Cleanup

The following existing code is superseded by this design and should be cleaned up during implementation:

- `src/foundry_mcp/core/batch_operations.py` — `_check_autonomous_limits()` (lines 557-637): Exported but has no callers. Guard logic (error threshold, context limit, blocked-task detection) is migrated into `core/autonomy/orchestrator.py`. Delete the function and its constants (`MAX_CONSECUTIVE_ERRORS`, `CONTEXT_LIMIT_PERCENTAGE`) after the orchestrator is functional and tested. **Important**: The helper `_check_all_blocked()` (lines 531-554) is also called by `prepare_batch_context()` (line 698) and must NOT be deleted. Either retain it in `batch_operations.py` or extract it into a shared utility (e.g., `core/task/_helpers.py`) that both `batch_operations` and the new orchestrator can import.
- `src/foundry_mcp/cli/context.py` — `AutonomousSession` dataclass (lines 33-84): Ephemeral session tracking replaced by durable session state. Retain `ContextSession` and `ContextTracker` for non-autonomy CLI session tracking (consultations, token usage). Remove the `autonomous` field from `ContextSession` after Phase F deprecation completes.

## Implementation Mapping

### Files to modify

- `src/foundry_mcp/tools/unified/task_handlers/handlers_query.py` — add `session` action handler with command dispatch
- `src/foundry_mcp/tools/unified/task_handlers/__init__.py` — register `session` and `session-step` actions
- `src/foundry_mcp/tools/unified/task_handlers/handlers_mutate.py` — enforce autonomy write-lock guard on protected task mutations
- `src/foundry_mcp/tools/unified/lifecycle.py` — enforce autonomy write-lock guard on protected lifecycle mutations
- `src/foundry_mcp/tools/unified/review.py` — add `fidelity-gate` action handler
- `src/foundry_mcp/core/discovery.py` — add autonomy capability flags and feature flag descriptors
- `mcp/capabilities_manifest.json` — add autonomy capabilities
- `src/foundry_mcp/core/batch_operations.py` — delete `_check_autonomous_limits` and related dead code; extract `_check_all_blocked` to shared utility (Phase B)
- `src/foundry_mcp/cli/context.py` — remove `autonomous` field from `ContextSession` (Phase F)

### Files to create

- `src/foundry_mcp/core/autonomy/__init__.py`
- `src/foundry_mcp/core/autonomy/models.py` — Pydantic models for session state, step results, gate policies, rebase results
- `src/foundry_mcp/core/autonomy/memory.py` — file-backed persistence (mirrors research memory pattern), active-session index management, lock file cleanup
- `src/foundry_mcp/core/autonomy/state_migrations.py` — schema versioning and migration functions (v1 initial, infrastructure for future migrations)
- `src/foundry_mcp/core/autonomy/orchestrator.py` — step engine, pause guards, feedback loop, spec integrity (metadata fast-path + phase-boundary re-hash), journal writes, opportunistic GC (outside lock scope)
- `src/foundry_mcp/core/autonomy/spec_hash.py` — spec structure hashing, mtime-based change detection, drift reporting, and rebase diff computation
- `src/foundry_mcp/core/autonomy/write_lock.py` — boolean write-lock check helpers and protected-action policy (v2: step-proof mint/verify/consume)
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py` — `session-step` action handler (`next`, `heartbeat`)

### Dependencies to add

- `python-ulid` — ULID generation for session and step IDs

### Documentation to update

- `docs/05-mcp-tool-reference.md` — new action reference for `session` and `session-step`
- `docs/03-workflow-guide.md` — autonomous workflow guide

## Testing Strategy

Tests are organized by rollout phase. **Phase A must-haves** are marked with **(P0)** — these gate the Phase A ship decision. Remaining tests land with their corresponding phases.

- Unit:
  - **(P0)** State schema validation and migration (v1 initial, forward path infrastructure)
  - **(P0)** Pause-guard decision logic (each guard independently and combined)
  - **(P0)** Active-session lookup resolution (single match, none, ambiguous)
  - **(P0)** One-active-session-per-spec enforcement
  - **(P0)** Feedback loop validation (missing result, mismatched step, all outcome types)
  - **(P0)** State transitions (start → pause → resume → end; start → complete; start → failed → reset)
  - **(P0)** Spec structure hash computation (deterministic across equivalent structures)
  - **(P0)** Spec integrity: drift detection (added/removed phases and tasks), metadata fast-path skip, phase-boundary re-hash
  - **(P0)** Resume context generation (truncation at 10 tasks, correct count, `files_touched` from journal, `journal_available` flag)
  - Stop-condition OR semantics (`context` safeguards, phase-complete stop, fidelity-cycle stop)
  - Gate policy evaluation (strict/lenient/manual for each verdict)
  - Active-session lookup excludes `reset` (explicit `session_id` required)
  - Atomic one-active-session-per-spec start enforcement under concurrent `start` calls
  - Idempotency key: duplicate `start` with same key returns existing session; different key returns `SPEC_SESSION_EXISTS`
  - Cursor encoding/decoding, sort stability, and invalid cursor handling (`INVALID_CURSOR`)
  - Gate attempt ID validation (binding, latest-attempt-wins, stale attempt rejection)
  - Multiple gate review attempts for the same `step_id`: latest attempt accepted, prior attempt rejected
  - Step staleness detection
  - Heartbeat staleness: warning-only when outstanding step is fresh; pause when idle between steps
  - Heartbeat grace window: first heartbeat allowed within `heartbeat_grace_minutes` of `created_at`
  - `stop_on_phase_completion=true` pauses at phase boundary after gate pass; `false` continues to next phase
  - `auto_retry_fidelity_gate=true` causes failed strict/lenient gate outcomes to emit `address_fidelity_feedback`, then `run_fidelity_gate` after remediation success (until cycle cap)
  - `auto_retry_fidelity_gate=false` pauses failed strict/lenient gate outcomes with `gate_failed`
  - Protected task/lifecycle mutation with active session returns `AUTONOMY_WRITE_LOCK_ACTIVE`; `bypass_autonomy_lock=true` overrides
  - Fidelity-cycle heuristic pauses with `fidelity_cycle_limit` once `max_fidelity_review_cycles_per_phase` is reached; counter resets on phase advance
  - Session reset (state file deleted, index entry removed, spec task statuses preserved); rejects non-`failed` states
  - Manual gate resume requires acknowledgment (`MANUAL_GATE_ACK_REQUIRED` / `INVALID_GATE_ACK`)
  - Lock acquisition timeout returns `LOCK_TIMEOUT`
  - Journal: lifecycle events produce entries; `start` journal write failure rolls back session; mid-session journal failures do not block
  - Rebase: no-change, structural diff, completed-task removal guard, force override, invalid source state
  - Force resume from `failed` with `spec_structure_changed` returns `SPEC_REBASE_REQUIRED`
  - `_check_all_blocked` extraction: shared utility callable from both `batch_operations` and orchestrator
- Integration:
  - **(P0)** Full lifecycle: start → heartbeat → next loop → context pause → resume → complete
  - **(P0)** Phase boundary: gate pass advances, gate fail pauses, manual gate requires human resume
  - **(P0)** Spec drift: editing spec mid-session causes `failed` on next `next` call; recovery via `rebase`
  - **(P0)** Duplicate start: second `start` for same spec returns `SPEC_SESSION_EXISTS`; with `force=true` ends existing and starts new
  - **(P0)** Action split: `session` rejects `next`/`heartbeat` commands; `session-step` rejects lifecycle commands
  - Phase-complete mode: gate pass pauses session with `phase_complete`, resume continues into next phase
  - Auto gate retry mode: failed strict/lenient verdicts trigger `address_fidelity_feedback` + re-review retries and eventually pause at `fidelity_cycle_limit`
  - Bypass resistance: direct `task complete`/status mutation during active session is rejected; `bypass_autonomy_lock=true` overrides
  - Fidelity retry path: repeated `review(action="fidelity-gate")` on same step issues new `gate_attempt_id`; `next` accepts latest attempt and rejects older attempt
  - GC: expired sessions cleaned (including index entries and orphaned lock files); opportunistic GC on `start` outside lock scope
  - Passive staleness: `status`/`list` return derived `effective_status` without mutating state
  - Maintenance sweep: stale `running` sessions are persisted to `paused`
  - Spec metadata unchanged: `next` call skips re-hash outside phase boundaries (verified via mock/spy on hash function)
  - Idempotent start: retry with same `idempotency_key` returns existing session
  - Session recovery: `reset` on corrupt session allows fresh start
  - Timeout paths: `next` and `fidelity-gate` timeout return `TIMEOUT` with actionable reconciliation fields
  - Lock timeout: concurrent `start` calls where lock is held returns `LOCK_TIMEOUT`
  - Backward compat: `session-config` `auto_mode=true` with explicit `spec_id` delegates to `session` start; without `spec_id` returns `AUTO_MODE_SPEC_REQUIRED`; `auto_mode=false` with active session delegates to `session` pause; with no active session returns `NO_ACTIVE_SESSION`
  - Journal trail: `resume_context.journal_hint` points to correct spec journal; `resume_context.files_touched` populated from journal entries
- Contract:
  - response-v2 envelope conformance for all new/extended actions
  - `session list` emits cursor pagination in `meta.pagination`
  - Capability manifest and schema alignment (includes `autonomy_sessions`, `autonomy_fidelity_gates`)
  - Error codes (`STEP_RESULT_REQUIRED`, `STEP_MISMATCH`, `AUTONOMY_WRITE_LOCK_ACTIVE`, `INVALID_GATE_EVIDENCE`, `INVALID_CURSOR`, `NO_ACTIVE_SESSION`, `AMBIGUOUS_ACTIVE_SESSION`, `SPEC_SESSION_EXISTS`, `SESSION_UNRECOVERABLE`, `LOCK_TIMEOUT`, `TIMEOUT`, `MANUAL_GATE_ACK_REQUIRED`, `INVALID_GATE_ACK`, `INVALID_STATE_TRANSITION`, `AUTO_MODE_SPEC_REQUIRED`, `SPEC_REBASE_REQUIRED`, `REBASE_COMPLETED_TASKS_REMOVED`, `HEARTBEAT_STALE`) in standard error envelope
- Stress:
  - Concurrent `next` calls on the same session: file locking prevents corruption, second caller gets `STEP_MISMATCH`
  - Rapid start/end cycles for the same spec: one-session-per-spec constraint holds under contention
  - Concurrent `start` with lock contention: one succeeds, others get `LOCK_TIMEOUT` or `SPEC_SESSION_EXISTS`
  - Large spec (100+ tasks): `next` response time remains bounded, `resume_context` truncation works correctly
  - GC under lock contention: opportunistic GC does not increase lock hold time

## Consequences

Positive:

- Enables durable autonomous progression with deterministic, verified gates.
- Closed feedback loop ensures orchestrator state reflects actual execution outcomes.
- Resume context enables meaningful continuation after context window resets.
- Aligns Foundry MCP contract with `claude-foundry` skill expectations.
- Reuses proven persistence/migration patterns already present in deep research.
- Gate policies provide flexibility across different risk profiles without sacrificing safety defaults.
- Automatic strict/lenient gate retries reduce manual babysitting while preserving deterministic stop behavior via cycle limits.
- Autonomy write-lock prevents accidental direct task/phase status mutations from bypassing gate sequencing.
- Configurable phase-completion stopping enables deliberate "one phase per run" execution without losing session continuity.
- Split `session`/`session-step` actions keep dispatch depth consistent with the rest of the codebase (two levels, not three) while separating concerns by performance profile.
- Spec integrity validation prevents silent state corruption from mid-session spec edits.
- `rebase` command provides a graceful recovery path for intentional spec edits without discarding session state — the common case of "fix something and continue" no longer requires `reset` + `start`.
- One-session-per-spec constraint eliminates the primary concurrent-execution failure mode.
- Journal integration provides a queryable audit trail for session history and enriches resume context.
- Idempotency keys on `start` make the create path safe for network retries.
- Opportunistic GC (outside lock scope) ensures cleanup happens even in short-lived CLI invocations without consuming lock budget.

Tradeoffs:

- Adds state-machine complexity and cross-tool coordination (`session` manages lifecycle, `session-step` returns step, caller executes, `review` evaluates gate, caller reports back to `session-step`).
- Two actions instead of one means callers must learn which action to use for which operation. Mitigated by clear naming (`session` = lifecycle, `session-step` = execution) and `next_action_hint` in responses.
- Feedback loop adds one extra round-trip per step vs. fire-and-forget, but eliminates state drift.
- Requires callers to adopt the `last_step_result` protocol — existing simpler integrations cannot use `session-step` `command="next"` without updating.
- Session cleanup TTLs are best-effort — opportunistic GC on `start` plus maintenance sweeps mitigates but does not guarantee immediate reclamation.
- Spec integrity checks add I/O. Mitigated by metadata fast-path (`mtime` + file size) with forced re-hash at phase boundaries.
- `files_touched` in step results is caller-reported and advisory. It is written to journal entries (not session state) and surfaced in `resume_context` for convenience.
- Added per-spec lock + index transaction for atomic start enforcement introduces extra lock contention during bursty `start` traffic. Lock-acquisition timeout (5s) bounds worst-case wait.
- Read-path purity shifts stale-to-paused persistence to mutating calls/maintenance loops, so persisted status may briefly lag derived status.
- Journal writes on every lifecycle event add I/O. Mitigated by best-effort semantics — journal failures do not block session operations.
- Heartbeat-based context guard is cooperative — a misbehaving caller can report inaccurate context usage. Step staleness provides a hard backstop.
- Automatic gate retries can increase AI-review cost/latency when a phase repeatedly fails fidelity.
- Write-lock enforcement requires updates across existing mutating handlers and can temporarily break legacy automations that mutate task/lifecycle state directly. The `bypass_autonomy_lock=true` escape hatch mitigates this for manual intervention.
- Fidelity-cycle stop heuristics can pause legitimate but complex phases; callers may need to raise `max_fidelity_review_cycles_per_phase` for difficult specs.
- `rebase` adds a new state transition path that must be tested thoroughly, particularly around completed-task validation and counter adjustment.

## Open Questions

1. **Per-phase gate policy overrides**: Should v1 support per-phase policies, or is session-level sufficient? Current decision: session-level only, tracked for v2.
2. **Partial resume granularity**: Should resume support restarting from a specific task within a phase, or always from where it left off? Current decision: resume from last checkpoint only. The escape hatch for "restart from a specific task" is `end`/`reset` (to release autonomy write-lock), then manual task-status manipulation via `task(action="update")`, then starting a new session.
3. **One-session-per-spec relaxation**: The current hard constraint prevents legitimate workflows like re-implementing a later phase while an earlier session is paused. `rebase` partially addresses this (the user can edit the spec and continue), but doesn't allow truly parallel sessions. Potential v2 options: allow a new session when the existing one has been paused for more than N hours, or add a `supersede` mode that inherits completed-phase state from the old session. Current decision: hard constraint in v1, tracked for v2.
4. **Cryptographic step proofs and signed gate evidence (v2)**: v1 uses boolean write-lock and `gate_attempt_id` matching, which is sufficient for local single-caller trust. For multi-tenant or remote deployments, cryptographic step-proof tokens (with TTL, replay detection, and allowed-action-set binding) and HMAC-signed gate evidence tokens would strengthen the trust boundary. Current decision: deferred to v2 — implement only if/when the system moves beyond local-trust deployments.

## Best-Practice References Consulted

- `dev_docs/mcp_best_practices/README.md#L24-L82`
- `dev_docs/mcp_best_practices/01-versioned-contracts.md`
- `dev_docs/codebase_standards/mcp_response_schema.md`
- `dev_docs/mcp_best_practices/02-envelopes-metadata.md`
- `dev_docs/mcp_best_practices/03-serialization-helpers.md`
- `dev_docs/mcp_best_practices/04-validation-input-hygiene.md`
- `dev_docs/mcp_best_practices/05-observability-telemetry.md`
- `dev_docs/mcp_best_practices/07-error-semantics.md`
- `dev_docs/mcp_best_practices/08-security-trust-boundaries.md`
- `dev_docs/mcp_best_practices/09-spec-driven-development.md`
- `dev_docs/mcp_best_practices/11-ai-llm-integration.md`
- `dev_docs/mcp_best_practices/12-timeout-resilience.md`
- `dev_docs/mcp_best_practices/13-tool-discovery.md`
- `dev_docs/mcp_best_practices/15-concurrency-patterns.md`
- `dev_docs/codebase_standards/naming-conventions.md`
