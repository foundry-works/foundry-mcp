# Post-Review Tasks

Date: 2026-02-16
Source: Senior engineering review of `tyler/foundry-mcp-20260214-0833` branch against `_research/PLAN.md`
Status: All tasks implemented and tested (2026-02-16). C1-C3, S1-S5, H1-H4, M1-M4, T1-T5 complete.

## Prioritization Key

- **C** = Critical — address before production concurrent workloads
- **H** = High — address before broad deployment
- **M** = Medium — track for near-term follow-up
- **T** = Test gap — improves confidence, low risk to defer

---

## C1. Add Optimistic Locking to Session Mutations ✅ DONE

**Status:** Implemented 2026-02-16

**Problem:** Session pause, resume, end, and reset load state, check status, then mutate — without holding a lock or verifying the state version hasn't changed. The `state_version` field is incremented but never checked for conflicts on save. A concurrent actor can change status between load and save, and the second writer silently overwrites.

**Impact:** Corrupt session state under concurrent access (multiple supervisors, operator + runner race).

**What was done:**
1. Added `VersionConflictError` exception class to `memory.py`.
2. Added `expected_version` keyword parameter to `AutonomyStorage.save()`. When provided, verifies on-disk `state_version` matches before writing. Raises `VersionConflictError` on mismatch. Default `None` skips check (full backward compat).
3. Added `VERSION_CONFLICT` to `ErrorCode` enum in `responses.py`.
4. Added `_save_with_version_check()` helper to `handlers_session.py` to avoid repeating try/catch 7 times.
5. Modified all 7 mutation save sites: pause, resume, end, rebase (no-change path), rebase (structural-change path), heartbeat, gate-waiver. Each captures `pre_mutation_version` before `state_version += 1`, then passes it to the version-checked save.
6. Added 4 tests to `test_memory.py`: correct version succeeds, wrong version raises, no-version skips check, concurrent race produces exactly one conflict.

---

## C2. Fail Rebase When Backup Hash Is Missing and Completed Tasks Exist ✅ DONE

**Status:** Implemented 2026-02-16

**Problem:** When the backup spec for structural diff computation is not found, the code creates an empty `StructuralDiff()`. This means `removed_completed_tasks` is always empty, and the session can silently lose task completion history during rebase.

**Impact:** Progress tracking data loss during spec rebases when backups are unavailable.

**What was done:**
1. Added `REBASE_BACKUP_MISSING` to `ErrorCode` enum in `responses.py`.
2. Added guard in rebase handler: when backup spec is not found and `session.completed_task_ids` is non-empty, returns `REBASE_BACKUP_MISSING` error (type `CONFLICT`) with details including completed task count, first 10 IDs, and hint to use `force=true`.
3. When `force=true`, logs warning and proceeds (existing behavior).
4. When `completed_task_ids` is empty, proceeds without error (no data at risk).
5. Added `backup_missing` and `completed_tasks_at_risk` fields to the journal metadata on rebase.
6. Added 3 tests to `test_handlers_session.py` (`TestRebaseBackupGuard`): missing backup + completed tasks fails, force succeeds, no completed tasks succeeds.

---

## C3. Bound the Proof/Replay Record Store ✅ DONE

**Status:** Implemented 2026-02-16

**Problem:** `consume_proof_with_lock` adds records to the proof store but there is no TTL-based cleanup or maximum record count. Long-running sessions accumulate unbounded proof records.

**Impact:** Storage growth proportional to session lifetime. A session with thousands of steps accumulates thousands of proof records that are never cleaned up.

**What was done:**
1. Added `MAX_PROOF_RECORDS = 500` and `PROOF_TTL_SECONDS = 3600` constants to `memory.py`.
2. Added `_cleanup_proof_records()` method to `AutonomyStorage` with two-phase cleanup:
   - Phase 1: TTL eviction — removes records whose `grace_expires_at` is older than `PROOF_TTL_SECONDS` from now.
   - Phase 2: LRU eviction — if still over `MAX_PROOF_RECORDS`, sorts by `consumed_at` and keeps newest.
3. Integrated cleanup into `save_proof_record()` (before adding new record).
4. Integrated cleanup into `consume_proof_with_lock()` (before adding new record).
5. Added 4 tests to `test_memory.py` (`TestProofStoreBounds`): TTL removes expired, TTL preserves fresh, max records eviction, 600-insert bounded load test.

---

## H1. Consolidate Loop Signal Computation to a Single Attachment Point ✅ DONE

**Status:** Implemented 2026-02-16

**What was done:**
1. Created `attach_loop_metadata(response, *, overwrite=True)` in `_helpers.py` as the single canonical attachment point for loop signal computation.
2. `_attach_session_step_loop_metadata()` now delegates to `attach_loop_metadata(response, overwrite=False)` (preserves handler values).
3. `handlers_session_step.py`: removed `_attach_loop_fields` definition, replaced with alias to `attach_loop_metadata`. Removed redundant inline `derive_loop_signal` call from `_build_next_step_response`.
4. Injected `pause_reason` into serialized step response data so `attach_loop_metadata` can derive signals from the response dict alone.
5. All 500 existing tests pass without modification.

**Problem:** `loop_signal` is computed independently in three places:
1. Session responses: `handlers_session.py` (~L544)
2. Step responses: `handlers_session_step.py` (~L150)
3. Error attachment: `_helpers.py` (~L364)

All three call `derive_loop_signal()` but pass different parameters and attach the result at different points in the response construction. If any call site diverges (different parameter extraction, different attachment location), the signal becomes inconsistent.

**Impact:** Maintenance risk. Any change to loop signal logic must be replicated in three places.

**Location:**
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py`
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py`
- `src/foundry_mcp/tools/unified/task_handlers/_helpers.py`

**Fix:**
1. Extract a single `attach_loop_metadata(response, session)` function.
2. Call it as a post-processing step on every session/step response, after the handler produces the base response.
3. Remove inline signal computation from individual handlers.

**Acceptance criteria:**
- `loop_signal` and `recommended_actions` are attached in exactly one code path.
- All existing tests pass without modification.

---

## H2. Make Journal Write Failures Observable ✅ DONE

**Status:** Implemented 2026-02-16

**What was done:**
1. Added `_inject_audit_status(response, *journal_results)` helper to `handlers_session.py`. Sets `meta.audit_status` to `"ok"`, `"partial"`, or `"failed"` based on journal write outcomes.
2. Captured `_write_session_journal()` return values at all 8 response-producing call sites: start, pause, resume, end, rebase (no-change), rebase (success), reset, gate-waiver.
3. Each response now passes through `_inject_audit_status()` before being returned.
4. Promoted orchestrator `_emit_audit_event()` failures from `debug` to `warning` level.
5. Added 6 tests in `TestAuditStatusObservability`: ok on success, ok on pause, failed when journal write fails, partial/ok/failed helper unit tests.

**Problem:** Journal writes (audit trail, session lifecycle events) are best-effort. Failures are logged at `debug` or `warning` level but not surfaced to the caller. Operators cannot tell whether the audit trail is complete.

**Impact:** Compliance and debugging gap. An operator may assume the audit trail is authoritative when writes have silently failed.

**Location:**
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py` — all journal write sites
- `src/foundry_mcp/core/autonomy/orchestrator.py` — `_emit_audit_event()`

**Fix (choose one):**
1. **Response metadata approach:** Add `meta.audit_status: "ok" | "partial" | "failed"` to responses that trigger journal writes. Callers can check this field.
2. **Health endpoint approach:** Add `audit_health` to `server(action="capabilities")` response, tracking last-write success/failure timestamp.
3. At minimum, promote critical journal write failures from `debug` to `warning` and include the failure count in session-status responses.

**Acceptance criteria:**
- Operator can determine whether audit writes succeeded for a given operation.

---

## H3. Split Large Handler Files ✅ DONE

**Status:** Implemented 2026-02-16

**What was done:**

Part 1 — `handlers_session.py` (2,758 lines → 5 files, max 1,045 lines):
- `_session_common.py` (569 lines): Shared helpers, constants, response builders
- `handlers_session_lifecycle.py` (1,045 lines): start, pause, resume, end, reset
- `handlers_session_query.py` (565 lines): status, list, events, heartbeat
- `handlers_session_rebase.py` (727 lines): rebase, gate_waiver
- `handlers_session.py` (92 lines): Re-export shim for backward compatibility

Part 2 — `orchestrator.py` (2,255 lines → 2 files, max 1,204 lines):
- `orchestrator.py` (1,199 lines): StepOrchestrator — init, compute_next_step, validation, recording, spec integrity, staleness/pause guards
- `step_emitters.py` (1,204 lines): StepEmitterMixin — step determination (steps 11-17), phase/task helpers, gate policy, 6 step builders, OrchestrationResult, error constants

All 506 autonomy tests pass without modification. All import paths preserved via re-exports.

---

## H4. Log Config Provenance Per Setting ✅ DONE

**Status:** Implemented 2026-02-16

**What was done:**
1. Added `INFO`-level provenance log lines in `_load_env()` for all 4 security-relevant settings: `role`, `allow_lock_bypass`, `allow_gate_waiver`, `enforce_required_phase_gates`. Each logs the setting name, value, and source env var.
2. Added provenance logging to `apply_autonomy_posture_profile()`: logs the applied profile and source, plus per-field override logs when posture defaults differ from current values (e.g., `"autonomy_security.role overridden by unattended posture (autonomy_runner ← maintainer)"`).
3. Added 4 tests in `TestConfigProvenanceLogging`: role env var provenance, allow_lock_bypass provenance, posture profile provenance, env var override after posture provenance.

**Problem:** Config loads from TOML → user config → env vars, but there is no logging of which source provided which value. Posture profile application and env var overrides interact in non-obvious ways.

**Impact:** Operator confusion when config doesn't behave as expected. Hard to debug "why is my role X when I set Y?"

**Location:**
- `src/foundry_mcp/config.py` — `_load_from_env()`, `_apply_autonomy_posture_defaults()`

**Fix:**
1. Add `INFO`-level log lines during config loading: `"Config: autonomy_security.role = autonomy_runner (source: FOUNDRY_MCP_ROLE env var)"`.
2. When posture profile overrides an explicit TOML setting, log the override: `"Config: autonomy_security.allow_lock_bypass overridden by unattended posture (false ← true)"`.

**Acceptance criteria:**
- Startup logs show the provenance of every security-relevant setting.
- Operators can trace any setting to its source.

---

## M1. Add Cross-Field Model Validators ✅ DONE

**Status:** Implemented 2026-02-16

**What was done:**
1. Added 3 `@model_validator(mode="after")` methods to `models.py`: `SessionLimits.validate_heartbeat_ordering`, `StepProofRecord.validate_grace_after_consumed`, `AutonomousSessionState.validate_satisfied_gates_subset`.
2. Tests: `tests/unit/test_core/autonomy/test_model_validators.py` — 11 cases covering valid/invalid for each validator.

---

## M2. Enforce Deprecation Removal Targets ✅ DONE

**Status:** Implemented 2026-02-16

**What was done:**
1. Added `_check_deprecation_expired()` helper to `_helpers.py`. Parses `removal_target` date, returns hard error if past deadline. `FOUNDRY_MCP_ALLOW_DEPRECATED_ACTIONS=true` escape hatch. Unparseable targets fail-open.
2. Wired into `_normalize_task_action_shape()` to check before returning legacy deprecation metadata.
3. Tests: `tests/unit/test_core/autonomy/test_deprecation_enforcement.py` — 5 cases.

---

## M3. Bound `reason_detail` Parameter Length ✅ DONE

**Status:** Implemented 2026-02-16

**Problem:** User-supplied `reason_detail` in pause/end operations has no length limit. Could bloat responses and journals.

**What was done:**
1. Added `_REASON_DETAIL_MAX_LENGTH = 2000` constant and `_validate_reason_detail()` helper to `_helpers.py`.
2. Added validation guard in `handlers_session_lifecycle.py` for `session-end` and `session-reset` actions.
3. Added validation guard in `handlers_session_rebase.py` for `gate-waiver` action.
4. Added 5 tests in `test_reason_detail_validation.py`: within limit, at limit, exceeding limit, None, empty string.

---

## M4. Add Audit Logging for Authorization Denials ✅ DONE

**Status:** Implemented 2026-02-16

**Problem:** Authorization denials are rate-limited but not audit-logged. There is no record of denied actions for security investigation.

**What was done:**
1. Added `_log_authorization_denial()` helper to `authorization.py` emitting structured WARNING logs with event, role, action, denied_action, required_role, and reason fields.
2. Wired into both `AuthzResult(allowed=False)` return paths in `check_action_allowed()`: unknown role (reason=unknown_role) and action not in allowlist (reason=action_not_in_allowlist).
3. Added 3 tests in `test_authorization_audit.py`: denial logs warning, allowed does not log, unknown role logs with reason.

---

## T1. Add Parametrized Loop Signal Mapping Exhaustiveness Test ✅ DONE

**Status:** Implemented 2026-02-16

**What was done:**
1. Created `test_loop_signal_exhaustive.py` with 36 parametrized rows covering every branch of `derive_loop_signal()`:
   - 1 PHASE_COMPLETE case
   - 2 SPEC_COMPLETE cases (status + pause_reason)
   - 14 BLOCKED_RUNTIME cases (one per `_BLOCKED_RUNTIME_ERROR_CODES` entry)
   - 2 BLOCKED_RUNTIME cases (repeated invalid gate evidence variants)
   - 4 FAILED cases (status, unrecoverable flag, two error codes)
   - 10 PAUSED_NEEDS_ATTENTION cases (one per `_PAUSED_NEEDS_ATTENTION_REASONS` entry)
   - 3 None/default cases (running, user pause, non-repeated gate evidence)

---

## T2. Add Step Proof Expiration Test with Time Advancement ✅ DONE

**Status:** Implemented 2026-02-16

**What was done:**
1. Added `TestStepProofExpiration` class to `test_memory.py` with 3 tests using `MagicMock(wraps=datetime)` to control `datetime.now()`:
   - Replay within 30s grace window succeeds (idempotent replay)
   - Replay past 30s grace window fails with PROOF_EXPIRED
   - Replay at exact grace boundary (30s) fails (strict greater-than check)

---

## T3. Add Verification Receipt Timing Boundary Tests ✅ DONE

**Status:** Implemented 2026-02-16

**What was done:**
1. Added `TestVerificationReceiptTimingBoundaries` class to `test_orchestrator.py` with 3 tests:
   - Receipt at exact issuance time (==) is valid
   - Receipt just after issuance (+1s) is valid
   - Receipt before issuance (-1s) is invalid (returns "earlier" error)

---

## T4. Add GC-by-TTL Verification Test ✅ DONE

**Status:** Implemented 2026-02-16

**What was done:**
1. Added `TestGarbageCollectionByTTL` class to `test_memory.py` with 9 parametrized tests:
   - 3 tests: session within TTL-1d survives (completed/7d, ended/7d, failed/30d)
   - 3 tests: session past TTL+1d is expired (completed/7d, ended/7d, failed/30d)
   - 2 tests: non-terminal sessions (running, paused) never expire even at 365d
   - 1 test: bulk `cleanup_expired()` removes all 3 terminal sessions past TTL

---

## T5. Add Config Source Provenance Test ✅ DONE

**Status:** Implemented 2026-02-16

**What was done:**
1. Added `TestConfigEnvVarOverridesToml` class to `test_config_hierarchy.py` with 3 tests:
   - FOUNDRY_MCP_ROLE env var overrides TOML `role = "maintainer"` to `"autonomy_runner"`
   - FOUNDRY_MCP_AUTONOMY_SECURITY_ALLOW_LOCK_BYPASS env var overrides TOML `allow_lock_bypass = true` to `false`
   - Provenance log mentions setting name and env var source when override occurs

---

## Agent-Level Soft Boundary Hardening

The following tasks address gaps where the agent (Claude Code) executing the skill can bypass MCP-level enforcement using its native tools (Bash, Write, Edit). The MCP authorization model gates all MCP tool calls via the `autonomy_runner` allowlist (14 actions), but native Claude Code tools operate outside this boundary.

### Threat Model

The agent runs inside Claude Code with access to Bash, Write, Edit, Read, Glob, and Grep tools. These are not gated by MCP authorization. While the SKILL.md instructs the agent to only use the session-step protocol, prompt instructions are soft boundaries — the agent could deviate.

**What the MCP server DOES enforce (hard boundaries):**
- Role allowlist: `autonomy_runner` can only call 14 specific MCP actions.
- `allow_lock_bypass=false`: server rejects bypass requests regardless of agent input.
- `allow_gate_waiver=false`: server rejects waiver requests regardless of role.
- `enforce_required_phase_gates=true`: server blocks phase transitions without gate satisfaction.
- Step proof one-time consumption: server validates proofs cryptographically.
- Receipt validation: server checks SHA-256 format, UTC timestamps, step binding.
- Spec integrity hash: server detects mid-session spec modification at next step boundary.

**What the MCP server does NOT enforce (soft boundaries):**
- Agent filesystem access (Write/Edit/Bash bypass MCP write lock).
- Agent git operations (commit/push outside session protocol).
- Agent config file modification (could change role/posture for future sessions).
- Agent journal/audit file manipulation (on-disk JSON, writable by agent).
- Agent spec file modification (detected eventually by hash check, but damage may be committed first).

---

### Implementation Summary (S1-S5)

**Files created:**
- `scripts/guard_autonomous_write.py` — Hook guard blocking Write/Edit to protected paths (specs, config, session state, journals, audit, proofs). Env vars: `FOUNDRY_GUARD_DISABLED`, `FOUNDRY_GUARD_EXTRA_BLOCKED`, `FOUNDRY_GUARD_EXTRA_ALLOWED`.
- `scripts/guard_autonomous_bash.py` — Hook guard blocking git write ops and shell writes to protected files. Env vars: `FOUNDRY_GUARD_DISABLED`, `FOUNDRY_GUARD_ALLOW_GIT_COMMIT`.

**Files modified:**
- `docs/guides/autonomy-agent-isolation.md` — Expanded from 142→375 lines. Added: threat model, guard script docs, hook config, four protection domain sections with layered mitigations, filesystem sandboxing guide, Docker example, post-session verification script, quick reference table.
- `skills/foundry-implement-v2/SKILL.md` — Added "Agent Isolation Constraints" section with filesystem/shell restrictions and allowed operations.
- `docs/guides/autonomy-supervisor-runbook.md` — Added "Agent Isolation Preflight" and "Post-Phase Integrity Verification" sections.

---

## S1. Document Agent-Environment Isolation Requirements ✅ DONE

**Status:** Implemented 2026-02-16

**Problem:** The SKILL.md documents MCP-level hardening but does not specify what the caller must do to constrain the agent's native tool access. Without filesystem isolation, the agent can bypass MCP enforcement.

**What was done:**
1. Enhanced `docs/guides/autonomy-agent-isolation.md` with comprehensive scope statement explaining that MCP authorization gates MCP tools only, and the caller is responsible for native tool constraints.
2. Added "Agent Isolation Constraints" section to `skills/foundry-implement-v2/SKILL.md` listing filesystem and shell restrictions enforced by guard scripts.
3. Added "Agent Isolation Preflight" section to `docs/guides/autonomy-supervisor-runbook.md` with required and recommended isolation checks.
4. Added "Post-Phase Integrity Verification" section to the supervisor runbook with concrete verification steps.
5. Created `scripts/guard_autonomous_write.py` and `scripts/guard_autonomous_bash.py` as reference implementations.
6. Documented recommended Claude Code hook configuration with environment variable controls.

---

## S2. Add Spec File Integrity Guard via Pre-Write Hook ✅ DONE

**Status:** Implemented 2026-02-16

**Problem:** The agent can modify the spec JSON file directly via the Write tool, bypassing the session protocol.

**What was done:**
1. `scripts/guard_autonomous_write.py` blocks Write/Edit targeting `specs/**/*.json`.
2. Documented a git pre-commit hook that blocks spec file commits when `FOUNDRY_AUTONOMOUS_SESSION=1` is set.
3. Added "Spec file integrity" section to isolation guide with layered mitigations: hook guard (preventive), pre-commit hook (preventive), MCP integrity check (detective), filesystem mount (high assurance).
4. Added Docker example showing `specs/` mounted read-only.

---

## S3. Add Config File Write Protection Guidance ✅ DONE

**Status:** Implemented 2026-02-16

**Problem:** The agent can write to `foundry-mcp.toml` to escalate privileges for future sessions.

**What was done:**
1. `scripts/guard_autonomous_write.py` blocks Write/Edit targeting `**/foundry-mcp.toml` and `**/.foundry-mcp.toml`.
2. `scripts/guard_autonomous_bash.py` blocks shell redirections and `sed -i` targeting config files.
3. Added "Config file protection" section to isolation guide with layered mitigations: hook guard, bash guard, read-only mount, startup integrity check with checksum comparison.
4. Added supervisor-side config checksum verification script.

---

## S4. Add Git Operation Guard for Unattended Runs ✅ DONE

**Status:** Implemented 2026-02-16

**Problem:** The agent can run `git commit`, `git push`, or `git reset` via Bash without orchestrator knowledge.

**What was done:**
1. `scripts/guard_autonomous_bash.py` blocks all git write operations by default (commit, push, reset, rebase, checkout, clean, merge, cherry-pick, revert).
2. Read-only git operations always allowed (status, diff, log, show, branch, etc.).
3. `FOUNDRY_GUARD_ALLOW_GIT_COMMIT=1` environment variable selectively enables `git commit` while keeping `git push` blocked — supervisor pushes after review.
4. Added "Git operation guarding" section to isolation guide with layered mitigations: bash guard, selective commit allowance, git credential scoping, pre-commit hook, post-phase review.
5. Added step-aware git commit allowance section to isolation guide and supervisor runbook.

---

## S5. Add Audit File Write Protection ✅ DONE

**Status:** Implemented 2026-02-16

**Problem:** Journal and audit ledger files are on-disk JSON writable by the agent.

**What was done:**
1. `scripts/guard_autonomous_write.py` blocks Write/Edit to `.foundry-mcp/journals/`, `.foundry-mcp/audit/`, `.foundry-mcp/proofs/`, and `.foundry-mcp/sessions/`.
2. `scripts/guard_autonomous_bash.py` blocks `rm` commands targeting audit directories.
3. Added "Audit file protection" section to isolation guide with layered mitigations: hook guard, bash guard, separate storage path, append-only filesystem attributes (`chattr +a`), integrity verification.
4. Documented MCP server storage path configuration for placing audit files outside agent workspace.
5. Added post-session integrity verification checklist for journal sequential timestamps and entry count.

---

## Suggested Execution Order

```
C1 (optimistic locking)  ─┐
C2 (rebase backup guard)  ├─ ✅ DONE — "Concurrency safety + rebase guard + proof bounds"
C3 (proof store bounds)   ─┘

S1 (isolation docs)       ─┐
S2 (spec write guard)     │
S3 (config write guard)   ├─ ✅ DONE — "Agent isolation guidance + hook examples"
S4 (git operation guard)  │
S5 (audit write guard)    ─┘

H1 (loop signal consolidation) ─┐
H2 (journal observability)      ├─ ✅ DONE — "Observability + DRY cleanup"
H4 (config provenance logging)  ─┘

H3 (file splitting)  ─── ✅ DONE — "Refactor large handler files" (mechanical, no behavior change)

M1 (model validators)              ─── ✅ DONE
M2 (deprecation enforcement)       ─── ✅ DONE

M3 (reason_detail bounds)          ─┐
M4 (auth denial audit logging)     ├─ ✅ DONE — "Remaining model hardening + test coverage"
T1-T5 (test coverage gaps)         ─┘
```
