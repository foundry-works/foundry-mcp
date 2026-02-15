# autonomy-boundary-hardening

## Mission

Harden the autonomous spec execution boundary so an orchestrating agent that is confused, stuck, or adversarial cannot silently bypass sequencing and quality controls.

## Objective

Upgrade autonomous execution from a cooperative workflow to an enforced trust boundary by constraining authority (P0), requiring proof-carrying progress (P1), and strengthening tamper resistance (P2). Each phase ships enforced with no feature flags or compatibility windows.

## Scope

### In Scope
- Fail-closed feature gating for all 13 session/session-step handler entrypoints
- Write-lock bypass restriction via config toggle (default off)
- Process-level role model (autonomy_runner / maintainer / observer) with action allowlists
- Role-based authorization at the unified dispatch layer
- Rate limiting on repeated authorization denials
- Escape hatch hardening (session-end, session-reset require maintainer + reason codes)
- Required phase-gate enforcement as server-side invariants (state model, computation, enforcement, waiver, audit, contract, integration test)
- One-time step proof tokens with idempotent replay
- Proof-carrying verification receipts
- Gate evidence integrity checksums (single-host HMAC)
- Independent gate-audit checker at terminal transitions
- Append-only hash-linked autonomy audit ledger
- Runtime isolation profiles for autonomy runner processes

### Out of Scope
- Fully compromised host with unrestricted filesystem/shell access
- Cryptographic trust across untrusted remote executors
- Per-request credential exchange (requires HTTP transport)
- Dual-control overrides requiring two distinct authenticated identities
- OS-level sandboxing (seccomp/AppArmor/chroot)

## Analysis Summary

### Current Codebase State

**Session Handlers** (`handlers_session.py`, 1460 LOC): `_feature_disabled_response()` exists (lines 92-104) but is called inconsistently. 9 handler entrypoints need gating. `force=true` flag on session-start/resume/rebase provides bypass paths.

**Session Step Handlers** (`handlers_session_step.py`, 492 LOC): 4 handler entrypoints. Step identity validation and gate evidence binding are strong. No feature gating helper exists — needs creation or import.

**Orchestrator** (`orchestrator.py`, 1534 LOC): 18-step priority sequence is well-structured. Phase completion at steps 12-17 has no required-gate invariant enforcement. `_evaluate_gate_policy()` supports STRICT/LENIENT/MANUAL but no mandatory gate per phase.

**Write Lock** (`write_lock.py`, 652 LOC): `bypass_flag=true` + `bypass_reason` allows bypass with audit logging. No config toggle to globally disable bypass. No role check on bypass path.

**Models** (`models.py`, 543 LOC): Schema version 2. No fields for required gate obligations or satisfaction tracking. `OverrideReasonCode` enum doesn't exist.

**Dispatch** (`common.py`, 213 LOC): `dispatch_with_standard_errors()` catches `ActionRouterError` and generic exceptions but has no authorization check point.

**Context** (`context.py`, 540 LOC): ContextVars for correlation_id, client_id, start_time, trace_context. No `server_role` context variable.

**Responses** (`responses.py`, 1693 LOC): 50+ error codes including autonomy-specific ones. No `AUTHORIZATION` or `RATE_LIMITED` error codes. Has `rate_limit_error()` helper.

**Authorization** (`authorization.py`): Does NOT exist yet — needs creation.

**Config** (`config.py`, 2444 LOC): Feature flags exist but `autonomy.role`, `autonomy.security.*`, and `autonomy.rate_limit.*` sections do not.

### Key Gaps
1. No centralized authorization module
2. Feature flag enforcement inconsistent across handlers
3. Write-lock bypass has no global disable or role restriction
4. No required-gate invariant enforcement at phase/spec completion
5. No step proof tokens or verification receipts
6. No audit ledger or runtime isolation

## Phases

### Phase 1: Constrain Authority (P0 Immediate Hardening)

**Purpose**: Ensure an orchestrating agent cannot unilaterally bypass policy just because it has general MCP access. Constrains who can do what.

**Tasks**:

1. **HB-01: Fail-closed feature gating** — Add `autonomy_sessions` feature-flag checks at all 13 session/session-step handler entrypoints. Make `feature_flags` a first-class `ServerConfig` field. Add or share `_feature_disabled_response` helper for `handlers_session_step.py`.
   - Files: `handlers_session.py`, `handlers_session_step.py`, `config.py`
   - Tests: `test_handlers_session.py`, `test_handlers_session_step.py` (`-k feature_disabled`)
   - Acceptance: Every handler returns `FEATURE_DISABLED` when off; no state created/mutated

2. **HB-02: Restrict write-lock bypass** — Add `autonomy.security.allow_lock_bypass` config (default `false`). When disabled, reject `bypass_autonomy_lock=true` regardless of caller. Emit `write_lock.bypass_denied` metric.
   - Files: `config.py`, `write_lock.py`, `_helpers.py`
   - Tests: `test_write_lock.py` (`-k bypass_denied_by_config`, `-k bypass_allowed_when_config_permits`)
   - Acceptance: Bypass rejected by default; denied attempts metricated

3. **HB-03: Authorization module + process-level role** — Create `authorization.py` with role resolution, action allowlist checking, `AuthzResult` dataclass. Add `autonomy.role` config with `FOUNDRY_MCP_ROLE` env override. Add `server_role_var` ContextVar.
   - Files: `authorization.py` (create), `config.py`, `context.py`, `server.py`
   - Tests: `test_authorization.py` (create)
   - Acceptance: Role enforcement works; env var overrides config; unconfigured defaults to observer

4. **HB-04: Integrate authorization into dispatch** — Wire `check_action_allowed()` into `dispatch_with_standard_errors()`. Add `AUTHORIZATION` error responses with recovery guidance. Error precedence: FEATURE_DISABLED → action validation → AUTHORIZATION → argument validation.
   - Files: `common.py`, `responses.py`
   - Tests: `test_dispatch_common.py`, `test_common.py`
   - Acceptance: Every dispatch path checks authorization; denied requests return proper error

5. **HB-05: Rate limiting on authorization denials** — Add `RateLimitTracker` to `authorization.py`. Track consecutive denials per action in sliding window. Return `RATE_LIMITED` with `retry_after` after threshold. Config: `autonomy.rate_limit.*`.
   - Files: `authorization.py`, `common.py`, `config.py`
   - Tests: `test_authorization.py` (`-k rate_limit*`)
   - Acceptance: Rate limiting triggers after N denials; resets on success or window expiry

6. **HB-06: Harden escape hatches** — Require `maintainer` role for `session-end`, `session-reset`, lock bypass. Add `OverrideReasonCode` enum (closed set). Require reason code for all privileged overrides.
   - Files: `handlers_session.py`, `models.py`, `write_lock.py`
   - Tests: `test_handlers_session.py`, `test_write_lock.py`
   - Acceptance: Non-privileged roles denied; reason code mandatory; bypass requires maintainer

7. **HB-07: Required-gate state model + migration** — Add per-phase required gate obligation/satisfaction fields to `AutonomousSessionState`. Bump schema to v3. Add migration for v2→v3.
   - Files: `models.py`, `state_migrations.py`, `conftest.py`
   - Tests: `test_context_tracker.py`, `test_memory.py`
   - Acceptance: Legacy v2 sessions migrate cleanly; new fields serialize/deserialize

8. **HB-08: Compute required gates at session start/rebase** — Derive required gates from spec phases during `session-start` and `session-rebase`. Preserve satisfied gates for unchanged phases on rebase.
   - Files: `handlers_session.py`, `models.py`
   - Tests: `test_handlers_session.py` (`-k required_gate`)
   - Acceptance: Every phase has required gate obligations after session creation/rebase

9. **HB-09: Enforce phase/spec completion invariants** — Block phase-complete and spec-complete transitions when required gates unsatisfied. Return machine-readable gate-block details with recovery guidance.
   - Files: `orchestrator.py`, `models.py`, `handlers_session_step.py`
   - Tests: `test_orchestrator.py`, `test_handlers_session_step.py`
   - Acceptance: Cannot emit PHASE_COMPLETE/COMPLETE_SPEC with unmet gates

10. **HB-10: Privileged gate-waiver path** — Add break-glass override for gate invariant failures. Restrict to `maintainer` + reason code. Default off (`allow_gate_waiver=false`).
    - Files: `handlers_session.py`, `models.py`, `authorization.py`
    - Tests: `test_handlers_session.py` (`-k gate_waiver`)
    - Acceptance: Non-privileged callers denied; waivers are auditable

11. **HB-12: Contract + observability for gate invariants** — Expose `required_phase_gates`/`satisfied_gates`/`missing_required_gates` in responses. Add config fields for enforcement toggles. Update discovery metadata.
    - Files: `handlers_session_step.py`, `handlers_session.py`, `discovery.py`, `config.py`
    - Tests: `test_handlers_session_step.py`, `test_handlers_session.py`
    - Acceptance: Clients detect missing gates from API data alone

12. **HB-13: End-to-end gate invariant test** — Integration test proving orchestrator cannot complete spec when required phase gate is skipped. Multi-phase fixture with gate bypass attempt + privileged waiver path.
    - Files: `test_integration.py`
    - Acceptance: Happy path requires gate completion; bypass attempts fail closed

**Verification**: Run full autonomy test suite. All 13 handlers return FEATURE_DISABLED when off. Role enforcement blocks unauthorized actions. Phase gates enforced at completion boundaries.

### Phase 2: Proof-Carrying Progress (P1)

**Purpose**: Move from claim-based progress ("I succeeded") to verifiable progress. Constrains what can be claimed.

**Tasks**:

1. **P1.1: One-time step proof tokens** — Add `step_proof` to step model. Require same token in `last_step_result`. Consume exactly once. Support idempotent re-submission within 30s grace window. Reject same proof with different payload as `PROOF_CONFLICT`.
   - Files: `models.py`, `orchestrator.py`, `memory.py`, `handlers_session_step.py`
   - Tests: Unit (proof mismatch/replay/idempotent/conflict), Integration (concurrent submissions, restart persistence)
   - Acceptance: Replay fails; idempotent resubmit works; concurrent safe

2. **P1.2: Verification receipts** — Server-issued receipt (hash of command/result metadata) for `execute_verification`. Success outcome requires valid receipt.
   - Files: `orchestrator.py`, verification tools, `handlers_session_step.py`
   - Tests: Unit (receipt required/invalid), Integration
   - Acceptance: Success cannot be claimed without receipt

3. **P1.3: Gate evidence signature** — HMAC integrity checksum on `pending_gate_evidence` keyed by server secret. Verify during gate outcome consumption. Auto-generate 32-byte secret on first start.
   - Files: `orchestrator.py`, `models.py`, server secret provisioning
   - Tests: Unit (modified evidence rejected), Integration
   - Acceptance: Tampered gate evidence rejected

4. **HB-11/P1.4: Independent gate-audit checker** — Recompute required gate obligations from spec + phase model. Compare with recorded evidence before phase-close and spec-complete transitions.
   - Files: `orchestrator.py`, `models.py`
   - Tests: `test_orchestrator.py` (`-k gate_audit`)
   - Acceptance: Tampered/missing records block terminal transitions

**Verification**: All step transitions require valid proof tokens. Verification success requires receipt. Gate evidence integrity verified. Independent audit checker catches inconsistencies.

### Phase 3: Tamper Resistance + Operations (P2)

**Purpose**: Make policy evasion operationally difficult, visible, and recoverable by humans. Constrains how safely the system operates under stress or misuse.

**Tasks**:

1. **P2.1: Append-only autonomy event ledger** — Hash-linked audit log for state transitions (step issued/consumed, pause/resume, bypass/override, reset/end). Each entry: timestamp, role, instance ID, action, payload digest. CLI `foundry audit verify` + auto-verify on session-start.
   - Files: New audit module, CLI extension, session handlers
   - Tests: Unit (hash chain integrity, break detection), Integration
   - Acceptance: Post-hoc verification detects tampering; reports exact divergence point

2. **P2.2: Runtime isolation profiles** — `workspace_root` path restriction for runner role. Tool permission filtering at session init. Shell execution PATH restriction + stdin-blocking subprocess timeout. Path traversal rejection.
   - Files: Authorization module, session init, verification execution
   - Tests: Unit (path traversal, tool filtering), Integration
   - Acceptance: Runner cannot escape workspace; traversal rejected; blocking subprocesses killed

**Verification**: Audit ledger integrity verifiable via CLI. Runner process cannot access files outside workspace root. All hardening active simultaneously.

## Dependencies

- HB-04 depends on HB-03 (authorization module must exist before dispatch integration)
- HB-05 depends on HB-04 (rate limiting integrates at dispatch level)
- HB-06 depends on HB-03 (escape hatch policy uses role model)
- HB-09 depends on HB-07, HB-08 (enforcement needs state model + computation)
- HB-10 depends on HB-09, HB-03 (waiver needs enforcement + role model)
- HB-12 depends on HB-09 (contract updates need enforcement logic)
- HB-13 depends on HB-09, HB-10, HB-12 (integration test needs all gate pieces)
- P1.4/HB-11 depends on HB-09 (independent checker complements enforcement)
- P2.2 depends on HB-03 (isolation profiles use runner role scoping)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing autonomy sessions mid-flight | High | Schema migration (v2→v3) with sensible defaults; legacy sessions get required-gate defaults applied |
| Confused agents retry-storming denied endpoints | Medium | Rate limiting (HB-05) with configurable thresholds and `retry_after` guidance |
| Maintainer role too powerful without audit trail | Medium | All privileged actions require reason codes (closed enum) and journal entries |
| Legacy specs missing phase gate declarations | High | Hard-fail with remediation via `foundry spec migrate-gates`; preflight audit before rollout |
| Step proof grace window timing side-channel | Low | Acceptable in single-host stdio model; documented for future multi-host adaptation |
| Feature flag not first-class in config | Medium | Promote to proper `ServerConfig` field as part of HB-01 |

## Success Criteria

- [ ] All 13 session/session-step handlers return FEATURE_DISABLED when autonomy_sessions is off
- [ ] Write-lock bypass rejected by default across all mutation routes
- [ ] Three-role model (autonomy_runner/maintainer/observer) enforced at dispatch level
- [ ] Unauthorized mutation attempts blocked with actionable recovery guidance
- [ ] Rate limiting triggers on repeated denials with retry_after
- [ ] session-end/session-reset require maintainer role + structured reason code
- [ ] Every phase has required gate obligations; completion blocked when unsatisfied
- [ ] Step proof tokens consumed exactly once; replays fail deterministically
- [ ] Verification success requires server-issued receipt
- [ ] Gate evidence integrity protected by HMAC
- [ ] Independent gate-audit checker blocks terminal transitions on mismatch
- [ ] Append-only audit ledger with hash-chain integrity verification
- [ ] Runner isolated to configured workspace root with path traversal rejected
- [ ] Zero successful state transitions from stale/replayed submissions
- [ ] Incident recovery possible for maintainer-role operators with auditable overrides

## Assumptions

- Single process, single host deployment for this plan version
- Local filesystem trusted for durability but not cryptographic immutability
- MCP server runs over stdio transport — no HTTP layer, no per-request credentials
- Caller identity determined by process-level config at startup, immutable for server lifetime
- Each phase ships enforced — no feature flags or compatibility windows
