# Post-Review Tasks

Date: 2026-02-16
Source: Senior engineering review of `tyler/foundry-mcp-20260214-0833` branch against `_research/PLAN.md`
Status: Draft — prioritized, not yet decomposed into specs

## Prioritization Key

- **C** = Critical — address before production concurrent workloads
- **H** = High — address before broad deployment
- **M** = Medium — track for near-term follow-up
- **T** = Test gap — improves confidence, low risk to defer

---

## C1. Add Optimistic Locking to Session Mutations

**Problem:** Session pause, resume, end, and reset load state, check status, then mutate — without holding a lock or verifying the state version hasn't changed. The `state_version` field is incremented but never checked for conflicts on save. A concurrent actor can change status between load and save, and the second writer silently overwrites.

**Impact:** Corrupt session state under concurrent access (multiple supervisors, operator + runner race).

**Location:**
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py` — pause (~L1485-1511), resume (~L1573-1620), end (~L1751), reset (~L2358)
- `src/foundry_mcp/core/autonomy/memory.py` — `save()` method

**Fix:**
1. Add `expected_version` parameter to `storage.save()`.
2. Before writing, verify the on-disk `state_version` matches `expected_version`.
3. Return a `VERSION_CONFLICT` error if mismatched — caller retries with fresh load.
4. Alternative: hold a per-session lock for the full mutation cycle (load → check → mutate → save), not just the save.

**Acceptance criteria:**
- Concurrent pause + resume on the same session produces a conflict error, not silent overwrite.
- Existing single-actor behavior is unchanged.

**Tests to add:**
- Two-thread race: load session, both mutate, second save returns conflict.
- Retry-after-conflict succeeds with fresh load.

---

## C2. Fail Rebase When Backup Hash Is Missing and Completed Tasks Exist

**Problem:** When the backup spec for structural diff computation is not found, the code creates an empty `StructuralDiff()`. This means `removed_completed_tasks` is always empty, and the session can silently lose task completion history during rebase.

**Impact:** Progress tracking data loss during spec rebases when backups are unavailable.

**Location:**
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py` — rebase logic (~L2061)

**Fix:**
1. If backup spec is not found and `session.completed_task_ids` is non-empty, fail the rebase (or require `force=true`).
2. Log the missing backup as a warning even when `completed_task_ids` is empty.
3. When `force=true` without backup, record the lost-diff condition in the journal.

**Acceptance criteria:**
- Rebase with missing backup + completed tasks returns an error unless `force=true`.
- Force-rebase without backup records the condition in the audit trail.

**Tests to add:**
- Rebase with completed tasks + missing backup → error.
- Rebase with completed tasks + missing backup + `force=true` → succeeds with journal warning.
- Rebase with no completed tasks + missing backup → succeeds (no data at risk).

---

## C3. Bound the Proof/Replay Record Store

**Problem:** `consume_proof_with_lock` adds records to the proof store but there is no TTL-based cleanup or maximum record count. Long-running sessions accumulate unbounded proof records.

**Impact:** Storage growth proportional to session lifetime. A session with thousands of steps accumulates thousands of proof records that are never cleaned up.

**Location:**
- `src/foundry_mcp/core/autonomy/memory.py` — proof record storage

**Fix:**
1. Add a `max_proof_records` bound (e.g., 500) with LRU eviction of expired records.
2. Add a `proof_ttl` (e.g., 1 hour) after which consumed proof records are eligible for cleanup.
3. Run cleanup opportunistically on `consume_proof_with_lock` when record count exceeds threshold.

**Acceptance criteria:**
- Proof store size stays bounded regardless of session duration.
- Grace-window replays still work within the TTL.
- Expired proofs are cleaned up without operator intervention.

**Tests to add:**
- Insert 600 proof records → oldest 100 are evicted.
- Proof within TTL is still replayable; proof past TTL is cleaned up.

---

## H1. Consolidate Loop Signal Computation to a Single Attachment Point

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

## H2. Make Journal Write Failures Observable

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

## H3. Split Large Handler Files

**Problem:** `handlers_session.py` (2,630 lines, 31 functions) and `orchestrator.py` (2,255 lines) are difficult to review and maintain.

**Impact:** Review friction, merge conflict risk, cognitive load.

**Location:**
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py`
- `src/foundry_mcp/core/autonomy/orchestrator.py`

**Fix:**
- `handlers_session.py`: Split into `handlers_session_lifecycle.py` (start/pause/resume/end/reset), `handlers_session_query.py` (status/list/events), and `handlers_session_rebase.py`.
- `orchestrator.py`: Split step-emission logic (steps 11-17) into `step_emitters.py`; keep the main orchestration sequence in `orchestrator.py`.

**Acceptance criteria:**
- No file exceeds ~1,000 lines.
- All existing tests pass without modification.
- Import paths remain stable (re-export from `__init__.py` if needed).

---

## H4. Log Config Provenance Per Setting

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

## M1. Add Cross-Field Model Validators

**Problem:** Several model invariants are assumed but not enforced:
- `SessionLimits`: no check that `heartbeat_grace_minutes < heartbeat_stale_minutes`.
- `AutonomousSessionState`: no invariant that values in `satisfied_gates` are subsets of `required_phase_gates`.
- `StepProofRecord`: no check that `grace_expires_at > consumed_at`.

**Impact:** Invalid state can be constructed without error, leading to silent misbehavior.

**Location:**
- `src/foundry_mcp/core/autonomy/models.py`

**Fix:** Add `@model_validator` (Pydantic) or `__post_init__` checks for each invariant.

---

## M2. Enforce Deprecation Removal Targets

**Problem:** Legacy action names have `removal_target` strings (e.g., `"2026-05-16_or_2_minor_releases"`) but no mechanism to actually block them when the date is reached.

**Impact:** Legacy paths accumulate indefinitely. Migration never completes.

**Location:**
- `src/foundry_mcp/tools/unified/task_handlers/_helpers.py` — `_LEGACY_ACTION_DEPRECATIONS`

**Fix:**
1. Parse `removal_target` into a date.
2. After the target date, return a hard error instead of a deprecation warning.
3. Add a `FOUNDRY_MCP_ALLOW_DEPRECATED_ACTIONS=true` escape hatch for emergency rollback.

---

## M3. Bound `reason_detail` Parameter Length

**Problem:** User-supplied `reason_detail` in pause/end operations has no length limit. Could bloat responses and journals.

**Location:**
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py` — pause, end handlers

**Fix:** Truncate or reject `reason_detail` exceeding a reasonable limit (e.g., 2,000 characters).

---

## M4. Add Audit Logging for Authorization Denials

**Problem:** Authorization denials are rate-limited but not audit-logged. There is no record of denied actions for security investigation.

**Location:**
- `src/foundry_mcp/core/authorization.py` — `check_action_allowed()`

**Fix:** Emit an audit event on denial (action, role, timestamp). Use existing audit ledger infrastructure.

---

## T1. Add Parametrized Loop Signal Mapping Exhaustiveness Test

**Problem:** Individual loop signals are tested through various scenarios, but there is no single parametrized test that enumerates all error codes from the WS3 mapping table and verifies each maps to the correct `loop_signal`.

**Location:** New test file or addition to `tests/unit/test_core/autonomy/test_handlers_session_step.py`

**Fix:** Create a parametrized test with one row per mapping table entry from PLAN.md WS3.

---

## T2. Add Step Proof Expiration Test with Time Advancement

**Problem:** Step proof expiration is tested in `test_error_paths.py` but only with mocked conditions. No test advances time past the grace window and verifies the proof is rejected.

**Fix:** Add a test that creates a proof, advances mock time past `grace_expires_at`, and verifies `STEP_PROOF_EXPIRED`.

---

## T3. Add Verification Receipt Timing Boundary Tests

**Problem:** No test for receipts issued at the exact boundary of the acceptance window.

**Fix:** Test receipt issued at `window_open`, `window_open + 1s`, and `window_open - 1s`.

---

## T4. Add GC-by-TTL Verification Test

**Problem:** Session garbage collection by TTL exists but is not explicitly tested with mock time.

**Fix:** Create sessions, advance mock time past TTL, trigger GC, verify old sessions are removed.

---

## T5. Add Config Source Provenance Test

**Problem:** No test verifies that env var overrides take precedence over TOML values for security-sensitive settings.

**Fix:** Set both TOML and env var for `role`, verify env var wins. Verify posture profile override logging.

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

## S1. Document Agent-Environment Isolation Requirements

**Problem:** The SKILL.md documents MCP-level hardening but does not specify what the caller must do to constrain the agent's native tool access. Without filesystem isolation, the agent can bypass MCP enforcement.

**Impact:** Operators may assume `FOUNDRY_MCP_ROLE=autonomy_runner` provides complete enforcement when it only covers MCP tool calls.

**Fix:**
1. Add an "Agent Isolation" section to the SKILL.md and supervisor runbook documenting:
   - MCP authorization gates MCP tools only, not native Claude Code tools.
   - The caller should use Claude Code hooks to restrict Write/Edit to expected directories.
   - The caller should use Claude Code hooks to restrict Bash commands (no `git push`, no config modification).
   - Alternatively, run the agent in a sandboxed filesystem where config files and spec files are read-only mounts.
2. Document the recommended hook configuration for unattended runs.

**Acceptance criteria:**
- Operators understand the boundary between MCP enforcement and agent-level enforcement.
- A recommended hook configuration is documented and tested.

---

## S2. Add Spec File Integrity Guard via Pre-Write Hook

**Problem:** The agent can modify the spec JSON file directly via the Write tool, bypassing the session protocol. The orchestrator detects this at the next spec integrity check (hash mismatch → `SPEC_REBASE_REQUIRED`), but the agent could commit the tampered spec to git before the check fires.

**Impact:** Spec tampering between integrity checks. The hash check is a detective control, not a preventive one.

**Fix:**
1. Document a Claude Code hook that blocks Write/Edit operations targeting `specs/` directory files during autonomous sessions.
2. Alternatively, add a pre-commit git hook that validates spec file checksums against the session's recorded hash.
3. Consider making the orchestrator's spec integrity check happen before recording step outcomes (currently it runs at step 4 in the 18-step sequence, after outcome recording at step 3).

**Acceptance criteria:**
- Spec file writes outside the MCP protocol are either blocked or detected before commit.

---

## S3. Add Config File Write Protection Guidance

**Problem:** The agent can write to `foundry-mcp.toml` to change `role = "maintainer"` or `profile = "debug"`. The MCP server reads config at startup, so this wouldn't affect the current session, but it would escalate privileges for future sessions.

**Impact:** Privilege escalation across sessions. A compromised or misbehaving agent in session N could weaken enforcement for session N+1.

**Fix:**
1. Document that `foundry-mcp.toml` should be read-only during unattended operation.
2. Recommend a Claude Code hook blocking writes to `**/foundry-mcp.toml` and `**/.foundry-mcp.toml`.
3. Consider adding a startup integrity check: if posture is `unattended` and config file was modified since last startup, log a prominent warning.

**Acceptance criteria:**
- Config file modification by the agent is either blocked or detected at next startup.

---

## S4. Add Git Operation Guard for Unattended Runs

**Problem:** The agent can run `git commit`, `git push`, or `git reset` via Bash without orchestrator knowledge or approval.

**Impact:** Unreviewed code pushed to remote. Potential for committing tampered specs, skipping gate results, or force-pushing over reviewed work.

**Fix:**
1. Document a Claude Code hook that restricts git operations during autonomous sessions to a known set (e.g., `git status`, `git diff`, `git log` allowed; `git commit`, `git push`, `git reset` blocked or require confirmation).
2. If the skill legitimately needs to commit (after `implement_task`), the hook should allow commits only when a session-step is active and the step type is `implement_task` or `address_fidelity_feedback`.
3. Document the recommended hook in the supervisor runbook.

**Acceptance criteria:**
- Git write operations are gated during unattended runs.
- Legitimate commit operations during implementation steps are allowed.

---

## S5. Add Audit File Write Protection

**Problem:** Journal and audit ledger files are on-disk JSON. The agent can modify or delete them via Write/Bash, compromising the audit trail.

**Impact:** Audit trail integrity. An agent could cover its tracks by modifying journal entries.

**Fix:**
1. Document that audit/journal directories should be append-only or read-only for the agent process.
2. Recommend running the MCP server with audit files in a directory the agent cannot write to (separate from the workspace).
3. Consider adding an integrity checksum to journal entries (chained hash) so that tampering is detectable even if not prevented.

**Acceptance criteria:**
- Audit file tampering is either prevented or detectable.

---

## Suggested Execution Order

```
C1 (optimistic locking)  ─┐
C2 (rebase backup guard)  ├─ PR: "Concurrency safety + rebase guard"
C3 (proof store bounds)   ─┘

S1 (isolation docs)       ─┐
S2 (spec write guard)     │
S3 (config write guard)   ├─ PR: "Agent isolation guidance + hook examples"
S4 (git operation guard)  │
S5 (audit write guard)    ─┘

H1 (loop signal consolidation) ─┐
H2 (journal observability)      ├─ PR: "Observability + DRY cleanup"
H4 (config provenance logging)  ─┘

H3 (file splitting)  ─── PR: "Refactor large handler files" (mechanical, no behavior change)

M1-M4  ─── PR: "Model hardening + deprecation enforcement"

T1-T5  ─── PR: "Test coverage gaps" (can be interleaved with any of the above)
```
