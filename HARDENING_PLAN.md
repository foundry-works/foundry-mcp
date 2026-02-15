# Autonomy Boundary Hardening Plan

## Narrative Summary

This plan upgrades autonomous execution from a cooperative workflow to an enforced trust boundary in phases.

Phase P0 constrains authority first. It hardens obvious bypasses by enforcing fail-closed feature gating, disabling lock bypass by default, introducing per-principal action allowlists, and making destructive escape hatches (`session-end`, `session-reset`, lock bypass) privileged and auditable. The goal is to ensure an orchestrating agent can no longer unilaterally bypass policy just because it has general MCP access.

Phase P1 reduces reliance on caller honesty. It introduces proof-carrying progression: one-time step proof tokens for `last_step_result`, verification receipts required for successful verification claims, and integrity-protected gate evidence. The goal is to move from claim-based progress ("I succeeded") to verifiable progress.

Across P0 and P1, phase completion is promoted to a hard contract invariant: every phase must satisfy required gates before the server allows progression. This makes gate execution mandatory even if an orchestrator tries to skip it.

Phase P2 strengthens tamper resistance and operational safety. It adds append-only, hash-linked autonomy audit logs, dual-control for destructive overrides, and restricted runtime profiles for autonomy runner identities. The goal is to make policy evasion operationally difficult, visible, and recoverable by humans.

In short: P0 constrains who can do what, P1 constrains what can be claimed, and P2 constrains how safely the system can be operated under stress or misuse.

## Goal

Harden the autonomous spec execution boundary so an orchestrating agent that is confused, stuck, or adversarial cannot silently bypass sequencing and quality controls.

This plan targets the autonomous execution surfaces implemented in:
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py`
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py`
- `src/foundry_mcp/core/autonomy/orchestrator.py`
- `src/foundry_mcp/tools/unified/review.py`
- `src/foundry_mcp/core/autonomy/write_lock.py`

## Threat Model

### In Scope

1. An orchestrating agent with valid MCP access that attempts to:
- skip or fabricate step outcomes
- bypass write locks
- force session transitions to avoid gate checks
- claim success without running verification

2. Honest-but-faulty behavior:
- agent gets stuck and starts using escape hatches incorrectly
- retries out of order
- stale state handling mistakes

### Out of Scope (for this plan version)

1. A fully compromised host with unrestricted filesystem/shell access.
2. Cryptographic trust across untrusted remote executors (called out as follow-up).

## Current Boundary Strength (Baseline)

### Strong Today

1. Step sequencing integrity:
- non-initial calls require `last_step_result`
- step identity binding via `step_id` and `step_type`
- gate evidence binding via `gate_attempt_id` + step/phase checks

2. Spec structure drift detection and explicit `session-rebase` recovery path.

### Weak Today

1. Outcome honesty remains caller-reported for most steps (`success|failure|skipped`).
2. Write-lock bypass exists (`bypass_autonomy_lock=true`) and is available at mutation surfaces.
3. `session-end` / `session-reset` are valid escape hatches and can be abused for policy evasion.
4. `handlers_session.py` defines `_feature_disabled_response` for `autonomy_sessions`, but runtime gating is not consistently enforced at handler entrypoints.
5. No per-principal action allowlist; all authenticated callers share the same autonomy mutation power.
6. Required phase-gate execution is not enforced as a server-side invariant; it is still vulnerable to orchestrator omission.

## Design Principles

1. Enforce policy server-side, not via prompt instructions.
2. Move from "claim-based success" to "proof-carrying success" for critical steps.
3. Remove broad bypasses from default execution paths.
4. Preserve operational recovery for humans via explicit privileged roles.
5. Roll out in phases with compatibility windows and migration tooling.
6. Treat required phase gates as state-machine invariants, not advisory workflow steps.

## Phased Plan

## P0: Immediate Hardening (1 sprint)

### P0.1 Enforce feature-flag guardrails for autonomy handlers

Implement explicit gate checks in:
- `handlers_session.py` (`session-start`, `session-resume`, etc.)
- `handlers_session_step.py` (`session-step-next`, replay, heartbeat)

Behavior:
- If `autonomy_sessions` is disabled, return deterministic `FEATURE_DISABLED`.

Acceptance:
- direct calls to session/session-step actions fail closed when flag is off
- add unit coverage in `tests/unit/test_core/autonomy/test_handlers_session.py` and `tests/unit/test_core/autonomy/test_handlers_session_step.py`

### P0.2 Restrict lock bypass by default

In `src/foundry_mcp/core/autonomy/write_lock.py` and mutation handlers:
- add config toggle `autonomy.allow_lock_bypass` (default `false`)
- when disabled, reject `bypass_autonomy_lock=true` regardless of caller input

Acceptance:
- bypass rejected by default in all protected task/lifecycle mutation routes
- metrics increment on denied bypass attempts

### P0.3 Add principal role model + action allowlist

Introduce an authorization layer (new module, e.g. `src/foundry_mcp/core/authorization.py`):
- resolve caller role from API key identity
- enforce per-role allowlist at dispatch boundary

Suggested roles:
- `autonomy_runner`: session + session-step + fidelity-gate only
- `maintainer`: full mutation surfaces
- `observer`: read-only operations

Enforcement point:
- centralized wrapper in unified dispatch flow before action handler execution

Acceptance:
- autonomy runner cannot invoke direct task/lifecycle mutation actions
- maintainer can still perform manual recovery with explicit audit trail

### P0.4 Harden escape hatch policy

Require privileged role for:
- `task(action="session-end")`
- `task(action="session-reset")`
- any lock bypass path

Require structured reason code (enum) instead of free-text-only.

Acceptance:
- non-privileged principal receives `AUTHORIZATION` error
- privileged calls must include reason code and are audited

### P0.5 Enforce required phase gates as server invariants

At autonomy state-machine level:
- compute and persist `required_phase_gates` per phase during `session-start` / `session-rebase`
- block phase completion when required gates are unsatisfied
- block `spec-complete` while any phase has unsatisfied required gates

Gate policy:
- require at least one fidelity gate (`run_fidelity_gate`) per phase by default
- allow spec-level expansion to additional required gates, but not removal of minimum gate type

Controlled break-glass:
- optional privileged `gate-waiver` path with reason code and audit record
- waiver disabled by default and never available to `autonomy_runner`

Acceptance:
- orchestrator cannot complete any phase without satisfying required gates
- orchestrator cannot complete spec if any phase is missing required gate completion
- any waiver is explicit, role-restricted, and observable

## P1: Proof-Carrying Progress (2-3 sprints)

### P1.1 One-time step proof token

Extend step model:
- add `step_proof` to `NextStep` / `LastStepIssued`
- require same token in `last_step_result`
- consume token exactly once

Implementation touchpoints:
- `src/foundry_mcp/core/autonomy/models.py`
- `src/foundry_mcp/core/autonomy/orchestrator.py`
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py`

Acceptance:
- replay of old `last_step_result` fails deterministically
- mismatched or missing proof yields validation error

### P1.2 Proof-carrying verification success

For `execute_verification`:
- introduce server-issued verification receipt (hash of command/result metadata)
- `outcome="success"` requires valid receipt in `last_step_result`

Implementation options:
- extend verification tool output with receipt
- validate receipt in orchestrator before marking verification task complete

Acceptance:
- success cannot be claimed without receipt
- fallback compatibility mode behind feature flag during migration

### P1.3 Gate evidence signature (single-host mode)

Upgrade `pending_gate_evidence`:
- include integrity checksum keyed by server secret
- verify checksum during `run_fidelity_gate` outcome consumption

Note: This is still same-host trust, but prevents accidental/malicious tampering through malformed client payloads.

Acceptance:
- modified gate evidence rejected with explicit error

### P1.4 Add independent gate-audit verification

Add a server-side checker that recomputes required gate obligations from spec + phase model and compares them with recorded evidence before allowing terminal transitions.

Enforcement:
- run checker on phase-close and spec-complete transitions
- return deterministic blocking errors that identify unmet phase and gate type

Acceptance:
- malformed or missing gate evidence cannot be hidden by orchestrator state edits
- terminal transitions fail closed when independent checker finds unmet obligations

## P2: Stronger Tamper Resistance + Operations (3+ sprints)

### P2.1 Append-only autonomy event ledger

Add append-only log for state transitions:
- include hash chain (`prev_hash`, `event_hash`)
- events: step issued, step consumed, pause/resume, bypass/override, reset/end

Store under:
- `specs/.autonomy/audit/` (or configurable secure path)

Acceptance:
- post-hoc integrity verification tool detects tampering

### P2.2 Dual-control for destructive overrides

Require dual approval token for:
- session reset/end on active incidents
- lock bypass in production mode

Acceptance:
- single agent identity cannot unilaterally force policy-evading transitions

### P2.3 Runtime isolation profile for orchestrating agents

Operational hardening:
- run autonomy runner with restricted key and reduced tool permissions
- deny filesystem writes outside workspace/spec-managed paths
- deny shell escapes in orchestrator runtime profile

Acceptance:
- policy-evasion paths blocked at credential/runtime layer, not just handler logic

## Config and Contract Changes

## New Config (proposed)

Add `[autonomy.security]`:
- `allow_lock_bypass = false`
- `require_reason_code_for_override = true`
- `enforce_step_proof = false` (rollout flag)
- `enforce_verification_receipts = false` (rollout flag)
- `restrict_session_end_reset_to_privileged = true`
- `enforce_required_phase_gates = true`
- `allow_gate_waiver = false`

Add `[authz]`:
- `enabled = true`
- `default_role = "observer"`
- role-action allowlists

## Response Contract Updates (proposed)

1. `next_step` includes `step_proof` (when enabled).
2. override responses include structured `override_policy` metadata.
3. authorization failures return explicit `error_type="authorization"` and role/action details.
4. phase state includes required-gate status metadata (`required_phase_gates`, `satisfied_gates`, `missing_required_gates`).
5. blocked transitions return machine-readable gate-block details (phase id, gate type, blocking reason).

## Test Plan

## Unit

1. Session/session-step fail closed when feature flag disabled.
2. Role-based allowlist enforcement at dispatcher.
3. Bypass denied when config disallows it.
4. Step proof mismatch/replay rejection paths.
5. Verification receipt required for success (when enabled).
6. Phase-close/spec-complete blocked when required phase gates are unsatisfied.
7. Gate waiver denied for non-privileged roles and denied globally when `allow_gate_waiver=false`.

## Integration

1. Autonomy runner can complete happy path without privileged actions.
2. Runner cannot mutate task/lifecycle directly.
3. Maintainer can execute controlled override with reason code.
4. Reset/end denied to runner role.
5. Runner cannot advance across any phase boundary without completed required gate evidence.
6. Spec completion is denied if any phase has unmet gate requirements.

## Property/Fuzz

1. Randomized step order / repeated step submissions cannot advance state.
2. Random payload tampering on gate evidence/step proof always rejected.
3. Randomized attempts to skip phase gates cannot produce phase-complete/spec-complete transitions.

## Rollout Strategy

1. Ship P0 with compatibility defaulting to secure behavior (`allow_lock_bypass=false`) but temporary opt-out for migration.
2. Enable required phase-gate invariants by default in P0; provide migration report tooling for legacy specs that lack phase gate declarations.
3. Ship P1 behind feature flags:
- `autonomy_step_proofs`
- `autonomy_verification_receipts`
4. Gather metrics and move flags to default-on after burn-in.
5. Remove legacy bypass paths and compatibility branches in a cleanup release.

## Success Metrics

1. `task.autonomy_write_lock.bypass.total` trends to near-zero.
2. Unauthorized mutation attempts blocked and observable.
3. Zero successful state transitions from stale/replayed step submissions.
4. Verification success claims without receipt reduced to zero when enforcement enabled.
5. Incident recovery still possible for privileged operators with auditable overrides.
6. Zero successful phase-complete/spec-complete transitions with unsatisfied required phase gates.

## Open Decisions

1. Principal identity source:
- API key mapping only, or richer auth context.

2. Step proof format:
- opaque random nonce vs HMAC payload token.

3. Receipt granularity:
- command-level receipt only vs richer artifact checks (stdout hash, exit code, test IDs).

4. Backward-compat timeline:
- how long to allow proof-optional mode.

5. Legacy spec migration behavior:
- auto-insert missing phase gates vs hard-fail with remediation guidance.

## Recommended Execution Order

1. P0.1 feature flag enforcement
2. P0.3 role allowlist
3. P0.2 bypass lockdown
4. P0.4 privileged override policy
5. P0.5 required phase-gate invariants
6. P1.1 step proof tokens
7. P1.2 verification receipts
8. P1.3 gate evidence signature
9. P1.4 independent gate-audit checker
10. P2 ledger + dual control + runtime isolation

## Implementation Tickets (P0.5 + P1.4)

### Ticket HB-01: Add required-gate state model + migration

Scope:
- Introduce explicit required gate tracking fields in autonomy state models.
- Bump state schema and add migration for existing session files.

Files:
- `src/foundry_mcp/core/autonomy/models.py`
- `src/foundry_mcp/core/autonomy/state_migrations.py`
- `tests/unit/test_core/autonomy/test_context_tracker.py`
- `tests/unit/test_core/autonomy/test_memory.py`
- `tests/unit/test_core/autonomy/conftest.py`

Implementation:
- Add model fields for per-phase required gate obligations and satisfaction tracking.
- Keep defaults fail-closed for required fidelity gate per phase.
- Bump `_schema_version` and add migration function to populate new fields for old sessions.
- Update test factories to include new default fields.

Acceptance tests:
- `pytest tests/unit/test_core/autonomy/test_context_tracker.py -k "migration"`
- `pytest tests/unit/test_core/autonomy/test_memory.py -k "schema_version or model_dump_by_alias"`

Done when:
- Loading legacy v2 session payloads migrates cleanly to new schema with required-gate defaults.
- New session states serialize/deserialize with required-gate fields intact.

### Ticket HB-02: Compute required phase gates at session start/rebase

Scope:
- Compute and persist `required_phase_gates` during `session-start` and `session-rebase`.
- Ensure spec changes reconcile required gate obligations without silently dropping required gates.

Files:
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py`
- `src/foundry_mcp/core/autonomy/models.py`
- `tests/unit/test_core/autonomy/test_handlers_session.py`

Implementation:
- Add a shared helper in `handlers_session.py` to derive required gates from spec phases.
- Invoke helper on `session-start` and `session-rebase` after loading current spec structure.
- On rebase, preserve satisfied gates for unchanged phases and mark new/changed requirements unsatisfied.

Acceptance tests:
- `pytest tests/unit/test_core/autonomy/test_handlers_session.py -k "session_start and required_gate"`
- `pytest tests/unit/test_core/autonomy/test_handlers_session.py -k "session_rebase and required_gate"`

Done when:
- Every phase has required gate obligations persisted immediately after session creation/rebase.
- Rebase does not allow required gate obligations to disappear due to spec drift.

### Ticket HB-03: Enforce phase/spec completion invariants in orchestrator

Scope:
- Block phase-complete and spec-complete transitions when required gates are unsatisfied.
- Return deterministic blocking errors with phase/gate metadata.

Files:
- `src/foundry_mcp/core/autonomy/orchestrator.py`
- `src/foundry_mcp/core/autonomy/models.py`
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py`
- `tests/unit/test_core/autonomy/test_orchestrator.py`
- `tests/unit/test_core/autonomy/test_handlers_session_step.py`

Implementation:
- Add invariant checks before `_create_pause_result(...PHASE_COMPLETE...)` and `_create_complete_spec_result(...)`.
- Add explicit orchestrator error code(s) for unsatisfied required gates.
- Extend `_map_orchestrator_error_to_response` to surface machine-readable gate-block details.

Acceptance tests:
- `pytest tests/unit/test_core/autonomy/test_orchestrator.py -k "required_gate and (phase_complete or complete_spec)"`
- `pytest tests/unit/test_core/autonomy/test_handlers_session_step.py -k "required_gate_block"`

Done when:
- Orchestrator cannot emit `PHASE_COMPLETE`/`COMPLETE_SPEC` while required gate obligations remain unmet.
- API responses identify the blocking phase/gate type deterministically.

### Ticket HB-04: Add privileged gate-waiver path (default off)

Scope:
- Add controlled break-glass override for required-gate invariant failures.
- Restrict waiver to privileged principals and structured reason codes.

Files:
- `src/foundry_mcp/tools/unified/task_handlers/__init__.py`
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py`
- `src/foundry_mcp/core/autonomy/models.py`
- `src/foundry_mcp/core/authorization.py` (from P0.3)
- `tests/unit/test_core/autonomy/test_handlers_session.py`

Implementation:
- Add a dedicated action path for gate waiver or equivalent override payload.
- Enforce role checks (`maintainer` only) and reason-code validation.
- Record waiver metadata on phase gate record and in session journal entries.
- Keep waiver globally disabled unless `allow_gate_waiver=true`.

Acceptance tests:
- `pytest tests/unit/test_core/autonomy/test_handlers_session.py -k "gate_waiver"`
- `pytest tests/unit/test_core/autonomy/test_handlers_session.py -k "authorization and gate_waiver"`

Done when:
- Non-privileged callers cannot waive required gates.
- Privileged waivers are explicit, reason-coded, and auditable.

### Ticket HB-05: Independent required-gate audit checker (P1.4 core)

Scope:
- Add server-side recomputation of required gate obligations and compare with recorded evidence before terminal transitions.

Files:
- `src/foundry_mcp/core/autonomy/orchestrator.py`
- `src/foundry_mcp/core/autonomy/models.py`
- `tests/unit/test_core/autonomy/test_orchestrator.py`

Implementation:
- Implement `_audit_required_gate_integrity(...)` that:
- rebuilds obligations from spec phases.
- validates each required gate has acceptable terminal evidence (`passed` or privileged `waived`).
- detects mismatches between obligation model and persisted phase gate records.
- Call checker immediately before phase-close and spec-complete transitions.

Acceptance tests:
- `pytest tests/unit/test_core/autonomy/test_orchestrator.py -k "gate_audit"`

Done when:
- Tampered/missing gate records block terminal transitions even if orchestrator flow attempts to proceed.
- Audit failures return deterministic error details.

### Ticket HB-06: Contract and observability updates for gate invariants

Scope:
- Expose required-gate status and gate-block errors in API contract and telemetry.
- Ensure discoverability and rollout controls are explicit.

Files:
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py`
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py`
- `src/foundry_mcp/core/discovery.py`
- `src/foundry_mcp/config.py`
- `tests/unit/test_core/autonomy/test_handlers_session_step.py`

Implementation:
- Add config fields for `enforce_required_phase_gates` and `allow_gate_waiver`.
- Add response metadata fields for `required_phase_gates`, `satisfied_gates`, `missing_required_gates`.
- Add structured error details for gate-blocked transitions.
- Expose capability/flag metadata for clients during rollout.

Acceptance tests:
- `pytest tests/unit/test_core/autonomy/test_handlers_session_step.py -k "gate_block_details"`
- `pytest tests/unit/test_core/autonomy/test_handlers_session.py -k "required_gate_status"`

Done when:
- Clients can reliably detect missing required gates from API data alone.
- Rollout flags are configurable and surfaced in capabilities metadata.

### Ticket HB-07: End-to-end invariant enforcement test

Scope:
- Add integration test to prove orchestrator cannot complete a spec when a required phase gate is skipped.

Files:
- `tests/unit/test_core/autonomy/test_integration.py`

Implementation:
- Build a multi-phase spec fixture.
- Execute session progression through `session-step-next/report` flow.
- Attempt to bypass gate completion and assert hard block.
- Then satisfy/waive gate (privileged path) and assert progression succeeds.

Acceptance tests:
- `pytest tests/unit/test_core/autonomy/test_integration.py -k "required_phase_gate"`

Done when:
- Happy path requires gate completion per phase.
- Bypass attempts fail closed and become visible in result metadata.
