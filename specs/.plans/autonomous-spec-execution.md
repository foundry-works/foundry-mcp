# Autonomous Spec Execution with Fidelity Gates and Resume

## Mission

Implement a durable Autonomous Session Controller (ASC) that provides deterministic orchestration for spec-driven execution — surviving interruption, enforcing gate semantics, preserving integrity under concurrency/failure, and supporting reliable resume from exact checkpoints.

## Objective

Introduce two new `task` actions — `task(action="session")` for lifecycle management and `task(action="session-step")` for the execution hot path — plus a `review(action="fidelity-gate")` action, with response-v2 envelopes. The ASC persists session state on disk with schema versioning, gates phase progression on fidelity reviews, supports configurable stop conditions, and enforces write-lock protection against bypass.

**Source**: ADR-002 (`dev_docs/architecture/adr-002-autonomous-spec-execution.md`)

## Review-Driven Design Decisions

These decisions address critical blockers identified during AI review of the initial plan:

1. **Atomic commit sequence for `start`** (addresses: atomicity + journal ordering). All state mutations during `start` happen inside the per-spec lock: acquire lock → check index → create session state (atomic write: temp+fsync+rename) → write journal entry (mandatory) → update active-session pointer → release lock → opportunistic GC. If journal write fails, rollback deletes the session state file before releasing the lock. No state is discoverable until the full commit sequence completes.

2. **Per-spec active-session pointer files instead of shared global index** (addresses: index locking strategy). Replace `spec_active_index.json` with per-spec pointer files: `specs/.autonomy/index/<spec_id>.active` containing the session_id. The per-spec lock is then sufficient — no global index lock needed. `list` scans the index directory. This eliminates concurrent-write races on a shared file.

3. **Idempotent `next` via step_id keying** (addresses: replay protection for hot path). The `next` command is made replay-safe by keying on `(session_id, last_step_result.step_id)`. If the caller retries with the same `step_id` in `last_step_result` and the session has already processed that step (i.e., `last_step_issued` has advanced), the orchestrator returns the cached `last_issued_response` (the next_step that was already issued). This provides exactly-once semantics without a separate request_id mechanism. The `last_issued_response` is stored in session state alongside `last_step_issued`.

4. **ADR as authoritative contract reference** (addresses: under-specified contract surface). The ADR contains exhaustive JSON examples for all request/response shapes, error codes with classification, and field-level documentation. The spec references ADR-002 sections as the authoritative contract source rather than duplicating them. Each error code maps to: `invalid_input` (bad request), `conflict` (state mismatch), `unavailable` (timeout/lock), or `not_found` per the ADR's existing error semantics.

5. **Crash-consistent atomic writes** (addresses: major suggestion on storage). All state file mutations use the atomic write pattern: write to temp file → `fsync` → rename over target. Load validates schema + required fields. Corrupt/unparseable files surface `state_corrupt` failure reason. Recovery path: `resume` with `force=true` or `reset`.

6. **Minimal write-lock enforcement in Phase A** (addresses: major suggestion on bypass window). Phase A includes `write_lock.py` with the basic `check_autonomy_write_lock()` guard, enforced on task mutation and lifecycle handlers from the start. This prevents the Phase A→B window where sessions exist but mutations are unprotected.

7. **Bypass guardrails** (addresses: major suggestion on bypass_autonomy_lock). `bypass_autonomy_lock=true` requires a `bypass_reason` string (mandatory when bypass is used). Both the bypass and reason are logged as warnings and written to a journal entry.

## Glossary and Routing Table

| Term | Definition |
|------|-----------|
| **Tool** | MCP tool entry point (e.g., `task`, `review`) |
| **Action** | First-level dispatch within a tool (e.g., `session`, `session-step`, `fidelity-gate`) |
| **Command** | Second-level dispatch within an action (e.g., `start`, `next`, `heartbeat`) |

| Tool | Action | Commands | Handler File |
|------|--------|----------|-------------|
| `task` | `session` | `start`, `status`, `pause`, `resume`, `rebase`, `end`, `list`, `reset` | `handlers_session.py` |
| `task` | `session-step` | `next`, `heartbeat` | `handlers_session_step.py` |
| `review` | `fidelity-gate` | *(single action, no sub-commands)* | `review.py` |

**Caller flow**: `task(action="session-step", command="next")` → dispatches to `handlers_session_step.py` → calls `orchestrator.py` step engine → may instruct caller to invoke `review(action="fidelity-gate")` → caller reports result back via next `task(action="session-step", command="next")`.

## Scope

### In Scope
- Persistent session state store with JSON file-backed storage, file locking, and schema versioning/migrations
- `task(action="session")` lifecycle commands: `start`, `status`, `pause`, `resume`, `end`, `list`, `reset`
- `task(action="session-step")` hot-path commands: `next`, `heartbeat`
- `review(action="fidelity-gate")` with gate-attempt evidence binding
- Gate policies: strict (default), lenient, manual — with auto-retry and fidelity-cycle stop heuristic
- Spec integrity validation via structure hashing (SHA-256) with mtime fast-path
- Autonomy write-lock enforcement on protected task/lifecycle mutations
- Closed feedback loop: callers report step outcomes before requesting next step
- Heartbeat and step staleness detection with configurable thresholds
- Configurable stop conditions (phase completion, context exhaustion, error threshold, task limit, fidelity-cycle limit)
- Resume context generation for post-context-window-reset continuation
- `task(action="session", command="rebase")` for accepting spec edits mid-session
- One-active-session-per-spec enforcement with atomic lock + index
- Idempotency keys on `start` for safe retries
- Journal integration for audit trail (lifecycle events, step outcomes, gate evaluations)
- Opportunistic GC for expired sessions (outside lock scope)
- Server capability flags: `autonomy_sessions`, `autonomy_fidelity_gates`
- Backward compatibility for `session-config` `auto_mode` during deprecation window
- ULID-format IDs for sessions and steps (via `python-ulid`)

### Out of Scope
- Code generation/editing — Foundry MCP provides orchestration, not implementation
- Cryptographic step-proof tokens and signed gate evidence (deferred to v2 for multi-tenant)
- Per-phase gate policy overrides (tracked for v2)
- Partial resume from specific task within a phase (v2)
- Relaxation of one-session-per-spec constraint (v2)
- Network filesystem support for session storage

## Phases

### Phase A: Persistent Session Store + Lifecycle Commands

**Purpose**: Establish the durable state layer and lifecycle management surface. This is the foundation — nothing else works without reliable persistence and state transitions.

**Tasks**:

1. **Add `python-ulid` dependency** (`implementation`, `pyproject.toml`)
   - Add `python-ulid` to project dependencies
   - Acceptance: dependency resolves and ULID generation works

2. **Create `core/autonomy/__init__.py`** (`implementation`, `src/foundry_mcp/core/autonomy/__init__.py`)
   - Package init with public API exports
   - Acceptance: clean import of core autonomy symbols

3. **Create `core/autonomy/models.py` — Pydantic models** (`implementation`, `src/foundry_mcp/core/autonomy/models.py`)
   - `AutonomousSessionState` model matching ADR schema (all fields from ADR section "Autonomous Session State Schema")
   - Enums: `SessionStatus` (running/paused/completed/failed/ended), `PauseReason` (12 values), `FailureReason` (4 values), `StepType` (6 values), `GatePolicy` (strict/lenient/manual), `GateVerdict` (pass/fail/warn), `PhaseGateStatus` (pending/passed/failed/waived), `StepOutcome` (success/failure/skipped)
   - `LastStepIssued`, `PendingGateEvidence`, `PendingManualGateAck`, `SessionCounters`, `SessionLimits`, `StopConditions`, `SessionContext`, `PhaseGateRecord` sub-models
   - `LastStepResult` request model with validation (step_id, step_type, task_id, outcome, note, files_touched, phase_id, gate_attempt_id)
   - `NextStep` response model with `step_id`, `type`, `instructions` array
   - `ResumeContext` model (spec_id, spec_title, active_phase_id, completed_task_count, recent_completed_tasks capped at 10, completed_phases, pending_tasks_in_phase, last_pause_reason, journal_available, journal_hint)
   - `RebaseResult` model (rebase_result enum, structural diff)
   - Validation: `context_usage_pct` 0-100, thresholds positive, `max_fidelity_review_cycles_per_phase >= 1`, counters non-negative, `idempotency_key` max 128 chars alphanumeric/hyphens/underscores
   - Acceptance: all models validate correctly, reject invalid inputs, serialize/deserialize round-trip

4. **Create `core/autonomy/state_migrations.py` — schema versioning** (`implementation`, `src/foundry_mcp/core/autonomy/state_migrations.py`)
   - Mirror deep-research pattern (`core/research/state_migrations.py`)
   - `CURRENT_SCHEMA_VERSION = 1`, `SCHEMA_VERSION_KEY = "_schema_version"`
   - `get_schema_version()`, `set_schema_version()`, `validate_state_version()`
   - `migrate_state()` with sequential migration application and recovery
   - Migration registry `MIGRATIONS: dict[(from_ver, to_ver)]` — empty for v1, infrastructure ready
   - Acceptance: version validation works, migration infrastructure exercises correctly with mock migrations

5. **Create `core/autonomy/memory.py` — file-backed persistence** (`implementation`, `src/foundry_mcp/core/autonomy/memory.py`)
   - Mirror `FileStorageBackend` pattern from `core/research/memory.py`
   - Storage paths: `specs/.autonomy/sessions/`, fallback `~/.foundry-mcp/autonomy/sessions/`
   - **Per-spec active-session pointer files**: `specs/.autonomy/index/<spec_id>.active` (contains session_id string). Eliminates shared global index — per-spec lock is sufficient for atomicity.
   - Per-spec locks: `specs/.autonomy/locks/spec_<spec_id>.lock`
   - File locking via `filelock` with 5-second lock-acquisition timeout
   - **Atomic writes**: all state file mutations use temp file → `fsync` → rename pattern. Load validates schema + required fields; corrupt files surface `state_corrupt`.
   - `save_session()`, `load_session()`, `delete_session()`, `list_sessions()` with pagination (cursor-based, `updated_at DESC, session_id DESC`)
   - Active-session pointer: `get_active_session_for_spec()`, `set_active_session()`, `remove_active_session()`. `lookup_active_session(workspace)` scans index directory for non-terminal sessions (deterministic: single match → use it, none → NO_ACTIVE_SESSION, multiple → AMBIGUOUS_ACTIVE_SESSION).
   - Cursor: encodes `(schema_version, last_updated_at, last_session_id, filters_hash)`. Rejects mismatched filters with INVALID_CURSOR.
   - Opportunistic GC: `cleanup_expired_sessions()` with configurable TTLs (completed/ended: 7 days, failed: 30 days), including pointer file removal and orphaned lock file cleanup. Lock file cleanup is conservative — only removes files whose corresponding session no longer exists (no PID checking in v1).
   - ID sanitization (same security pattern as research memory)
   - Acceptance: CRUD operations work, file locking prevents corruption, pointer management is atomic within lock scope, GC cleans expired sessions, atomic writes survive crash mid-write

6. **Register `session` and `session-step` actions in task handlers** (`implementation`, `src/foundry_mcp/tools/unified/task_handlers/__init__.py`)
   - Add `ActionDefinition` entries for `session` and `session-step`
   - Feature-flag guard: check `autonomy_sessions` flag, return error if disabled
   - Import handlers from new handler files
   - Acceptance: actions registered, dispatch works, feature flag gates access

7. **Create `task_handlers/handlers_session.py` — session lifecycle handler** (`implementation`, `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py`)
   - Command dispatch: `start`, `status`, `pause`, `resume`, `end`, `list`, `reset`
   - `start`: acquire per-spec lock (5s timeout, LOCK_TIMEOUT on failure) → check pointer file for existing session → idempotency_key check → read spec + compute spec_structure_hash → create session state (atomic write: temp+fsync+rename) → write journal entry (mandatory — if fails, delete state file and release lock without creating pointer) → write active-session pointer file → release lock → opportunistic GC outside lock
   - `status`: read-only, compute effective_status with staleness metadata (stale_reason, stale_detected_at) without mutating state
   - `pause`: transition running → paused with user pause_reason, write journal
   - `resume`: validate pause_reason resolution, manual-gate acknowledgment check (MANUAL_GATE_ACK_REQUIRED/INVALID_GATE_ACK), force from failed (re-validate spec, SPEC_REBASE_REQUIRED if structure changed), write journal
   - `end`: transition running/paused/failed → ended, write journal
   - `list`: paginated with cursor (opaque, versioned, server-generated), status_filter, spec_id filter, limit (default 20, max 100), read-only with effective_status for stale sessions, INVALID_CURSOR on bad cursor
   - `reset`: requires explicit session_id (no active-session lookup), only valid from `failed` (INVALID_STATE_TRANSITION otherwise), deletes state file + index entry, write journal
   - Response format: response-v2 envelope with session data (matching ADR response schema)
   - Resume context: included on `resume` responses (ResumeContext model)
   - Acceptance: all commands work, state transitions are correct per ADR table, error codes match ADR, journal entries written

8. **Create `task_handlers/handlers_session_step.py` — session-step action stub** (`implementation`, `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py`)
   - Register `next` and `heartbeat` commands with feature-flag guard
   - `heartbeat`: update context_usage_pct, estimated_tokens_used, consecutive_error_delta, last_completed_task_id in session state
   - `next`: stub that returns FEATURE_NOT_READY (Phase B implements full step engine)
   - Acceptance: heartbeat updates state correctly, next returns clear "not yet implemented" error

9. **Create `core/autonomy/spec_hash.py` — spec structure hashing** (`implementation`, `src/foundry_mcp/core/autonomy/spec_hash.py`)
   - `compute_spec_structure_hash()`: SHA-256 of sorted phase IDs, task IDs, task-to-phase parent mappings, phase ordering
   - `get_spec_file_metadata()`: return mtime + file size
   - `compute_structural_diff()`: added/removed phases and tasks between two hash computations
   - Deterministic: equivalent structures produce identical hashes regardless of JSON key ordering
   - Acceptance: hash is deterministic, diff computation is correct, metadata retrieval works

10. **Add autonomy capability flags to discovery** (`implementation`, `src/foundry_mcp/core/discovery.py`)
    - Add `autonomy_sessions` and `autonomy_fidelity_gates` to `ServerCapabilities`
    - Feature flags with initial defaults: both `off`
    - Acceptance: capabilities reported in server info, feature flags queryable

11. **Update capabilities manifest** (`implementation`, `mcp/capabilities_manifest.json`)
    - Add `session` and `session-step` action entries to task tool
    - Add autonomy capability flags and feature flag descriptors
    - Document new parameters
    - Acceptance: manifest validates, new actions discoverable

12. **Create `core/autonomy/write_lock.py` — write-lock helpers** (`implementation`, `src/foundry_mcp/core/autonomy/write_lock.py`)
    - `check_autonomy_write_lock(spec_id, workspace, bypass_flag, bypass_reason)` → returns lock status
    - `is_protected_action(action_name)` → boolean for protected vs. read-only actions
    - Protected actions: task start/complete/update-status/block/unblock, lifecycle status mutations
    - Bypass: `bypass_autonomy_lock=true` requires `bypass_reason` string, logs warning + writes journal entry
    - Acceptance: protected actions blocked when session active, bypass works with reason, read-only unaffected

13. **Add write-lock guard to task mutation and lifecycle handlers** (`implementation`, `src/foundry_mcp/tools/unified/task_handlers/handlers_mutate.py`, `src/foundry_mcp/tools/unified/lifecycle.py`)
    - Check `check_autonomy_write_lock()` before protected mutations in both handlers
    - Return AUTONOMY_WRITE_LOCK_ACTIVE when blocked
    - Accept bypass_autonomy_lock=true + bypass_reason parameters
    - Acceptance: protected mutations rejected during active session, bypass overrides with audit trail

14. **Unit tests — Phase A P0** (`implementation`, `tests/unit/test_autonomy_models.py`, `tests/unit/test_autonomy_memory.py`, `tests/unit/test_autonomy_migrations.py`, `tests/unit/test_spec_hash.py`, `tests/unit/test_write_lock.py`)
    - State schema validation and migration (v1 initial, forward path infrastructure)
    - Active-session lookup resolution (single match, none, ambiguous) — using pointer file scan
    - One-active-session-per-spec enforcement
    - State transitions (start → pause → resume → end; start → complete; start → failed → reset)
    - Spec structure hash computation (deterministic across equivalent structures)
    - Resume context generation (truncation at 10 tasks, correct count)
    - Idempotency key: duplicate start returns existing session; different key returns SPEC_SESSION_EXISTS
    - Cursor encoding/decoding, sort stability, invalid cursor handling (filters_hash mismatch)
    - Lock acquisition timeout returns LOCK_TIMEOUT
    - Atomic write: crash mid-write leaves no partial state (temp file cleaned up)
    - Write-lock: protected mutations blocked during active session; bypass requires reason; read-only unaffected

15. **Integration tests — Phase A P0** (`implementation`, `tests/integration/test_autonomy_lifecycle.py`)
    - Full lifecycle: start → heartbeat → pause → resume → end
    - Duplicate start: second start for same spec returns SPEC_SESSION_EXISTS; with force=true ends existing
    - Action split: session rejects next/heartbeat; session-step rejects lifecycle commands
    - Write-lock: direct task complete during active session rejected; bypass_autonomy_lock=true with bypass_reason overrides

**Verification**:
- All unit tests pass (P0 set)
- Integration tests pass for lifecycle commands
- Feature flag gates all new actions
- Session state persists across simulated process restarts

### Phase B: Step Engine, Feedback Loop, and Enforcement

**Purpose**: Implement the core execution hot path — the step engine that drives autonomous task progression with deterministic guards and enforcement.

**Tasks**:

1. **Create `core/autonomy/orchestrator.py` — step engine** (`implementation`, `src/foundry_mcp/core/autonomy/orchestrator.py`)
   - Implement full `next` command logic following ADR orchestration rules (17-step priority order):
     1. **Replay detection**: if `last_step_result.step_id` matches the step that was already processed (i.e., `last_step_issued` has advanced past it), return cached `last_issued_response` for exactly-once semantics
     2. Validate feedback (STEP_RESULT_REQUIRED if missing on non-initial call)
     3. Validate step identity (STEP_MISMATCH if last_step_issued doesn't match)
     4. Record step outcome (mark task complete, increment errors, record gate, write journal)
     5. Validate spec integrity (mtime as optimization, re-hash on any ambiguity or phase boundary)
     6. Check terminal states
     7. Enforce step staleness (hard backstop)
     8. Enforce heartbeat staleness (cooperative signal with grace window)
     9. Enforce pause guards (context_limit, error_threshold, task_limit)
     10. Enforce fidelity-cycle stop heuristic
     11. Check all-blocked → pause
     12. Phase tasks complete + verifications pending → execute_verification
     13. Phase verifications complete + gate pending → run_fidelity_gate
     14. Gate fails policy → pause or address_fidelity_feedback (auto-retry)
     15. Fidelity feedback completed → run_fidelity_gate retry
     16. Gate passed + stop_on_phase_completion → pause (phase_complete)
     17. Gate passed + next task exists → implement_task
     18. No remaining tasks → complete_spec
   - ULID generation for step_id tokens
   - Persist `last_step_issued` AND `last_issued_response` (cached next_step for replay safety) before responding
   - Journal writes for all lifecycle transitions (best-effort except start)
   - Acceptance: orchestration rules match ADR exactly, all step types emitted correctly, replay of same step_id returns cached response

2. **Implement full `next` command in handlers_session_step.py** (`implementation`, `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py`)
   - Replace Phase A stub with orchestrator integration
   - Response: session_id, status, state_version, next_step (or null)
   - next_step includes step_id, type, task_id/phase_id, task_title, instructions array
   - Acceptance: full step engine works end-to-end

3. **Extract `_check_all_blocked()` to shared utility** (`refactoring`, `src/foundry_mcp/core/task/_helpers.py`)
   - Extract from `core/batch_operations.py` to shared location
   - Update imports in batch_operations.py
   - Import in orchestrator.py
   - Acceptance: both batch_operations and orchestrator use shared helper, no behavior change

4. **Delete `_check_autonomous_limits()` from batch_operations.py** (`refactoring`, `src/foundry_mcp/core/batch_operations.py`)
   - Remove dead function and constants (MAX_CONSECUTIVE_ERRORS, CONTEXT_LIMIT_PERCENTAGE)
   - Guard logic migrated to orchestrator.py
   - Retain `_check_all_blocked()` import from shared utility
   - Acceptance: no dead code, batch_operations still works, orchestrator has equivalent logic

5. **Unit tests — Phase B** (`implementation`, `tests/unit/test_orchestrator.py`)
   - Pause-guard decision logic (each guard independently and combined)
   - Feedback loop validation (missing result, mismatched step, all outcome types)
   - Replay safety: retried `next` with already-processed step_id returns cached response
   - Spec integrity: drift detection, metadata optimization skip, phase-boundary re-hash, re-hash on any ambiguity
   - Stop-condition OR semantics
   - Step staleness detection
   - Heartbeat staleness: warning-only when outstanding step fresh; pause when idle
   - Heartbeat grace window
   - stop_on_phase_completion=true pauses at phase boundary; false continues
   - _check_all_blocked extraction: shared utility callable from both modules

6. **Integration tests — Phase B P0** (`implementation`, `tests/integration/test_autonomy_step_engine.py`)
   - Full lifecycle: start → heartbeat → next loop → context pause → resume → complete
   - Phase boundary: gate pass advances, gate fail pauses
   - Spec drift: editing spec mid-session causes failed on next call
   - Spec metadata unchanged: next call skips re-hash outside phase boundaries
   - Replay: retrying last `next` call returns identical response without double-counting

**Verification**:
- All Phase A + B unit tests pass
- Integration tests exercise full step engine lifecycle
- Write-lock enforcement verified on all protected actions
- Spec integrity validation works with both fast-path and slow-path

### Phase C: Fidelity Gate Action and Gate Policy Enforcement

**Purpose**: Close the gate enforcement loop — the review tool creates gate evidence, the orchestrator validates and applies policy.

**Tasks**:

1. **Add `fidelity-gate` action to review tool** (`implementation`, `src/foundry_mcp/tools/unified/review.py`)
   - New action handler: `_handle_fidelity_gate()`
   - Request: spec_id, session_id, phase_id, step_id + existing fidelity inputs
   - Runs phase fidelity review (reuse existing review infrastructure)
   - Writes pending gate-attempt evidence to session state: `{gate_attempt_id, step_id, phase_id, session_id, verdict, issued_at}`
   - Multiple attempts for same step_id: latest replaces prior (latest-attempt-wins)
   - Response: spec_id, session_id, phase_id, step_id, gate_attempt_id, verdict, gate_policy echo, gate_passed_preview, review_path, findings
   - Feature-flag guard: check `autonomy_fidelity_gates`
   - Acceptance: gate evidence written correctly, verdict computed, response matches ADR schema

2. **Implement gate policy evaluation in orchestrator** (`implementation`, `src/foundry_mcp/core/autonomy/orchestrator.py`)
   - Gate evidence validation: match gate_attempt_id against pending_gate_evidence (session/phase/step binding)
   - INVALID_GATE_EVIDENCE on invalid/stale attempt
   - Policy application:
     - strict: pass only on verdict=pass
     - lenient: pass on pass or warn
     - manual: always pause with gate_review_required
   - Auto-retry: failed strict/lenient → address_fidelity_feedback → run_fidelity_gate retry (bounded by cycle cap)
   - Fidelity-cycle counter increment on each accepted gate result, reset on phase advance
   - Acceptance: gate policy correctly evaluated for all verdict+policy combinations, auto-retry cycle works

3. **Unit tests — Phase C** (`implementation`, `tests/unit/test_gate_policy.py`)
   - Gate policy evaluation (strict/lenient/manual for each verdict)
   - Gate attempt ID validation (binding, latest-attempt-wins, stale rejection)
   - Multiple gate review attempts for same step_id
   - auto_retry_fidelity_gate=true triggers address_fidelity_feedback + retry
   - auto_retry_fidelity_gate=false pauses with gate_failed
   - Fidelity-cycle heuristic pauses at limit; counter resets on phase advance

4. **Integration tests — Phase C** (`implementation`, `tests/integration/test_autonomy_gates.py`)
   - Phase boundary: gate pass advances, gate fail pauses, manual gate requires human resume
   - Auto gate retry: failed verdicts trigger remediation + re-review, eventually pause at cycle limit
   - Fidelity retry path: repeated reviews issue new gate_attempt_id; next accepts latest, rejects older
   - Manual gate resume requires acknowledgment (MANUAL_GATE_ACK_REQUIRED / INVALID_GATE_ACK)

**Verification**:
- Gate policy enforcement correct for all combinations
- Auto-retry cycle bounded by fidelity-cycle cap
- Manual gate acknowledgment flow works end-to-end
- Gate evidence binding prevents misattribution

### Phase D: Rebase Command

**Purpose**: Enable graceful recovery from intentional spec edits without discarding session state.

**Tasks**:

1. **Implement `rebase` command in session handler** (`implementation`, `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py`)
   - Valid from paused or failed states (INVALID_STATE_TRANSITION otherwise)
   - Re-read spec, compute new spec_structure_hash
   - No-change: return rebase_result="no_change", transition to running
   - Hash differs: compute structural diff (added/removed phases and tasks)
   - Validate completed tasks not removed (REBASE_COMPLETED_TASKS_REMOVED unless force=true)
   - Force: remove missing task IDs from completed_task_ids, decrement counters
   - Update hash, mtime, file_size in session state
   - Clear failure_reason/pause_reason, transition to running
   - Return RebaseResult with structural diff
   - Write journal entry
   - Resume context included in response
   - Acceptance: all rebase scenarios from ADR work correctly

2. **Unit tests — Phase D** (`implementation`, `tests/unit/test_rebase.py`)
   - Rebase: no-change, structural diff, completed-task removal guard, force override, invalid source state
   - Force resume from failed with spec_structure_changed returns SPEC_REBASE_REQUIRED

3. **Integration tests — Phase D** (`implementation`, `tests/integration/test_autonomy_rebase.py`)
   - Spec drift recovery via rebase: edit spec mid-session → failed → rebase → running
   - Rebase with additive changes preserves completed task history
   - Rebase with removed completed tasks: rejected without force, accepted with force

**Verification**:
- Rebase correctly handles all ADR scenarios
- Structural diff accurately reports changes
- Completed-task validation prevents accidental data loss

### Phase E: Skill + Hook Alignment

**Purpose**: Align the `claude-foundry` skill with the new session/session-step contract so the skill can drive autonomous execution.

**Tasks**:

1. **Research current claude-foundry skill session commands** (`investigation`)
   - Identify all session-related commands the skill currently expects (start/status/pause/resume/end)
   - Map to new session/session-step actions
   - Identify gaps or contract mismatches
   - Acceptance: complete mapping document

2. **Update claude-foundry skill session integration** (`implementation`)
   - Update skill to use `task(action="session")` for lifecycle
   - Update skill to use `task(action="session-step")` for execution
   - Implement feedback loop protocol (last_step_result reporting)
   - Implement heartbeat reporting
   - Handle resume context for post-context-window continuation
   - Acceptance: skill drives autonomous execution end-to-end

3. **Integration tests — Phase E** (`implementation`)
   - Skill-driven lifecycle: start → step loop → pause → resume → complete
   - Context window reset: skill resumes with resume_context

**Verification**:
- claude-foundry skill drives full autonomous lifecycle
- Feedback loop protocol correctly implemented
- Resume context enables meaningful continuation

### Phase F: Legacy Deprecation

**Purpose**: Clean up backward-compatibility shims after the deprecation window.

**Tasks**:

1. **Implement session-config backward compatibility** (`implementation`, `src/foundry_mcp/tools/unified/task_handlers/handlers_query.py`)
   - `auto_mode=true` → deprecation warning + delegate to session start (requires spec_id, AUTO_MODE_SPEC_REQUIRED if missing)
   - `auto_mode=false` → deprecation warning + delegate to session pause via active-session lookup (NO_ACTIVE_SESSION if none)
   - `get=true` → unchanged lightweight config read
   - Acceptance: backward compat works, deprecation warnings logged

2. **Remove `autonomous` field from ContextSession** (`refactoring`, `src/foundry_mcp/cli/context.py`)
   - Remove `autonomous` field from `ContextSession` dataclass
   - Retain `ContextSession` and `ContextTracker` for non-autonomy tracking
   - Acceptance: no references to removed field, non-autonomy tracking unaffected

3. **Remove `auto_mode` support from session-config** (`refactoring`, `src/foundry_mcp/tools/unified/task_handlers/handlers_query.py`)
   - Remove deprecation shim after two release cycles
   - `auto_mode` parameter returns clear "removed" error directing to session action
   - Acceptance: clean removal, no backward-compat code

4. **Integration tests — Phase F** (`implementation`, `tests/integration/test_autonomy_compat.py`)
   - Backward compat: auto_mode=true with spec_id delegates to session start; without spec_id returns AUTO_MODE_SPEC_REQUIRED
   - auto_mode=false with active session delegates to pause; no session returns NO_ACTIVE_SESSION

**Verification**:
- Backward compatibility works during deprecation window
- Clean removal after window closes
- No regressions in non-autonomy session tracking

## Contract Tests (Cross-Phase)

- response-v2 envelope conformance for all new/extended actions
- `session list` emits cursor pagination in `meta.pagination`
- Capability manifest and schema alignment (autonomy_sessions, autonomy_fidelity_gates)
- Error codes in standard error envelope: STEP_RESULT_REQUIRED, STEP_MISMATCH, AUTONOMY_WRITE_LOCK_ACTIVE, INVALID_GATE_EVIDENCE, INVALID_CURSOR, NO_ACTIVE_SESSION, AMBIGUOUS_ACTIVE_SESSION, SPEC_SESSION_EXISTS, SESSION_UNRECOVERABLE, LOCK_TIMEOUT, TIMEOUT, MANUAL_GATE_ACK_REQUIRED, INVALID_GATE_ACK, INVALID_STATE_TRANSITION, AUTO_MODE_SPEC_REQUIRED, SPEC_REBASE_REQUIRED, REBASE_COMPLETED_TASKS_REMOVED, HEARTBEAT_STALE

## Stress Tests (Cross-Phase)

- Concurrent `next` calls on same session: file locking prevents corruption, second caller gets STEP_MISMATCH
- Rapid start/end cycles for same spec: one-session-per-spec holds under contention
- Concurrent `start` with lock contention: one succeeds, others get LOCK_TIMEOUT or SPEC_SESSION_EXISTS
- Large spec (100+ tasks): next response time bounded, resume_context truncation works
- GC under lock contention: opportunistic GC doesn't increase lock hold time

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| State machine complexity leads to subtle bugs in transition logic | High | Exhaustive unit tests for every state transition; ADR provides authoritative transition table |
| File locking inadequate on network filesystems | Medium | Document local-fs-only requirement; provide fallback path config |
| Heartbeat-based context guard is cooperative/advisory | Medium | Step staleness provides hard backstop; document advisory nature |
| Write-lock enforcement breaks existing automations | Medium | bypass_autonomy_lock=true escape hatch; end/reset session first |
| Fidelity auto-retry increases AI-review cost | Low | Bounded by max_fidelity_review_cycles_per_phase; configurable per session |
| Spec integrity re-hash I/O on every step | Low | mtime fast-path skips re-hash when metadata unchanged; forced only at phase boundaries |
| GC on start adds latency | Low | GC runs outside lock scope; only processes expired sessions |
| Journal write failures | Low | Best-effort for mid-session events; only start journal is mandatory (rollback on failure) |

## Assumptions

- Local POSIX filesystem for session storage (network filesystems not supported for locking)
- `filelock` library available (already a dependency)
- Single-caller trust model sufficient for v1 (no multi-tenant requirements)
- Existing fidelity review infrastructure can be reused for fidelity-gate action
- `python-ulid` library is stable and suitable for ID generation
- Two release cycles is sufficient deprecation window for session-config auto_mode

## Dependencies

- `python-ulid` (new dependency)
- `filelock` (existing dependency)
- Existing: `core/research/memory.py` pattern, `core/research/state_migrations.py` pattern
- Existing: review tool fidelity infrastructure
- Existing: journal tool for audit trail
- Existing: spec loading and task query infrastructure

## Success Criteria

- [ ] Full autonomous lifecycle works: start → step loop → gate → complete
- [ ] Session state survives simulated process restart and context window reset
- [ ] Gate policies correctly enforce strict/lenient/manual semantics
- [ ] Spec edits mid-session detected and recoverable via rebase
- [ ] Write-lock prevents bypass of orchestration sequencing
- [ ] Resume context enables meaningful continuation after interruption
- [ ] One-session-per-spec enforced atomically under concurrency
- [ ] All ADR error codes implemented and tested
- [ ] Backward compatibility for session-config maintained during deprecation window
- [ ] All P0 unit and integration tests pass
- [ ] response-v2 envelope conformance for all new actions
- [ ] Journal audit trail captures all lifecycle events
