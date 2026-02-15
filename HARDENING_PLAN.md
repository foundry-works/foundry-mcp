# Autonomy Boundary Hardening Plan

## Narrative Summary

This plan upgrades autonomous execution from a cooperative workflow to an enforced trust boundary in phases.

Phase P0 constrains authority first. It hardens obvious bypasses by enforcing fail-closed feature gating, disabling lock bypass by default, introducing a process-level role with action allowlists, and making destructive escape hatches (`session-end`, `session-reset`, lock bypass) privileged and auditable. The goal is to ensure an orchestrating agent can no longer unilaterally bypass policy just because it has general MCP access.

Phase P1 reduces reliance on caller honesty. It introduces proof-carrying progression: one-time step proof tokens for `last_step_result`, verification receipts required for successful verification claims, and integrity-protected gate evidence. The goal is to move from claim-based progress ("I succeeded") to verifiable progress.

Across P0 and P1, phase completion is promoted to a hard contract invariant: every phase must satisfy required gates before the server allows progression. This makes gate execution mandatory even if an orchestrator tries to skip it.

Phase P2 strengthens tamper resistance and operational safety. It adds append-only, hash-linked autonomy audit logs and restricted runtime profiles for autonomy runner processes. The goal is to make policy evasion operationally difficult, visible, and recoverable by humans.

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

### Deployment Assumptions

1. Single process, single host deployment for this plan version.
2. Local filesystem is trusted for durability but not for cryptographic immutability.
3. The MCP server runs over **stdio transport** (stdin/stdout pipes). There is no HTTP layer and no per-request credential exchange. The server process and its invoking host process share a trust domain.
4. Caller identity is determined by **process-level configuration at startup**, not by per-request authentication. The server's role is immutable for its lifetime.

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
- network retries cause duplicate proof/receipt submissions
- tight retry loops on denied actions (requires rate limiting)

### Recovery Protocol Requirements

Every blocking error (gate unsatisfied, proof mismatch, authorization denied) MUST include machine-readable recovery guidance so an honest-but-confused orchestrator can self-correct without human intervention:
- `GATE_UNSATISFIED` → response includes the specific gate type(s) and phase ID(s) that need completion, plus the action to invoke.
- `PROOF_MISMATCH` / `PROOF_EXPIRED` → response indicates whether the step can be re-requested (via `session-step-next`) or requires escalation.
- `AUTHORIZATION` → response includes the required role and, when safe, the specific action that was denied.
- `RATE_LIMITED` → response includes retry-after interval.

This is a first-class contract requirement, not an afterthought — confused agents are the most common failure mode and unactionable errors will cause retry storms.

### Out of Scope (for this plan version)

1. A fully compromised host with unrestricted filesystem/shell access.
2. Cryptographic trust across untrusted remote executors (called out as follow-up).
3. Per-request credential exchange (requires HTTP transport; see Future Work).
4. Dual-control overrides requiring two distinct authenticated identities (requires per-request auth or multi-process coordination; see Future Work).

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
5. No role-based action restriction; all callers share the same autonomy mutation power regardless of whether the process is an autonomous runner or a human-driven session.
6. Required phase-gate execution is not enforced as a server-side invariant; it is still vulnerable to orchestrator omission.

## Design Principles

1. Enforce policy server-side, not via prompt instructions.
2. Move from "claim-based success" to "proof-carrying success" for critical steps.
3. Remove broad bypasses from default execution paths.
4. Preserve operational recovery for humans via explicit privileged roles.
5. Ship each phase enforced — no feature flags or compatibility windows.
6. Treat required phase gates as state-machine invariants, not advisory workflow steps.
7. Every blocking error must include actionable recovery guidance (see Recovery Protocol Requirements in Threat Model).
8. Be honest about the trust boundary. The identity model matches the transport reality (process-level config over stdio), not an aspirational architecture.

## Phased Plan

### P0: Immediate Hardening (1 sprint)

### P0.1 Ensure fail-closed behavior for disabled autonomy

The existing `autonomy_sessions` feature gate is not consistently enforced at handler entrypoints. Fix that.

Implement explicit gate checks in:
- `handlers_session.py` (`session-start`, `session-resume`, etc.)
- `handlers_session_step.py` (`session-step-next`, replay, heartbeat)

Behavior:
- If `autonomy_sessions` is disabled, return deterministic `FEATURE_DISABLED`.

Acceptance:
- direct calls to session/session-step actions fail closed when feature is off
- add unit coverage in `tests/unit/test_core/autonomy/test_handlers_session.py` and `tests/unit/test_core/autonomy/test_handlers_session_step.py`

### P0.2 Restrict lock bypass by default

In `src/foundry_mcp/core/autonomy/write_lock.py` and mutation handlers:
- add config toggle `autonomy.allow_lock_bypass` (default `false`)
- when disabled, reject `bypass_autonomy_lock=true` regardless of caller input

Acceptance:
- bypass rejected by default in all protected task/lifecycle mutation routes
- metrics increment on denied bypass attempts

### P0.3 Add process-level role model + action allowlist

Introduce authorization module (`src/foundry_mcp/core/authorization.py`):

Identity source: **startup configuration**. The server reads its role from config or environment at process start. The role is immutable for the server's lifetime. This matches the stdio transport model where the server runs as a subprocess of the host process and there is no per-request credential exchange.

```toml
[autonomy]
role = "autonomy_runner"  # or "maintainer" or "observer"
```

The role can also be set via environment variable `FOUNDRY_MCP_ROLE` (env takes precedence over config file, allowing the same config to be used with different roles in different contexts).

Role definitions:
- `autonomy_runner`: session + session-step + fidelity-gate actions only
- `maintainer`: full mutation surfaces (task, lifecycle, session, override)
- `observer`: read-only operations (spec view, task list, status queries)

Allowlist schema — flat `(role, action)` tuples:

```toml
[autonomy.roles.autonomy_runner]
allowed_actions = [
    "session-start", "session-resume", "session-step-next",
    "session-step-report", "session-heartbeat", "session-rebase",
    "run_fidelity_gate",
]

[autonomy.roles.maintainer]
allowed_actions = ["*"]

[autonomy.roles.observer]
allowed_actions = [
    "list", "get", "view", "status", "search", "progress",
]
```

Enforcement point:
- centralized check in unified dispatch flow after action normalization
- role loaded once at server init and stored in module-level state (not re-read per request)
- error precedence:
  - `FEATURE_DISABLED`
  - action validation (`unsupported action`, malformed action name)
  - `AUTHORIZATION`
  - argument validation

Rate limiting on denied actions:
- track consecutive `AUTHORIZATION` denials per action within a sliding window
- after N consecutive denials (configurable, default 10) within 60 seconds, begin returning `RATE_LIMITED` with a `retry_after` interval
- this prevents confused agents from hammering denied endpoints in tight loops
- rate limit state is in-memory only (resets on process restart, which is acceptable)

Acceptance:
- autonomy runner cannot invoke direct task/lifecycle mutation actions
- unconfigured role defaults to `observer` (fail-closed to read-only, not full deny)
- audit logs include the configured role and a server instance ID
- maintainer can still perform manual recovery with explicit audit trail
- denied actions emit `authz.denied` metrics with role and action labels
- repeated denials trigger rate limiting with `RATE_LIMITED` response and `retry_after`

### P0.4 Harden escape hatch policy

Require `maintainer` role for:
- `task(action="session-end")`
- `task(action="session-reset")`
- any lock bypass path (when `allow_lock_bypass=true`)

Require structured reason code (enum) instead of free-text-only for all privileged overrides.

Acceptance:
- `autonomy_runner` and `observer` roles receive `AUTHORIZATION` error for escape hatch actions
- `maintainer` calls must include reason code and are audited
- reason codes are a closed enum — free-text-only reason is rejected

### P0.5 Enforce required phase gates as server invariants

At autonomy state-machine level:
- compute and persist `required_phase_gates` per phase during `session-start` / `session-rebase`
- block phase completion when required gates are unsatisfied
- block `spec-complete` while any phase has unsatisfied required gates

Gate policy:
- require at least one fidelity gate (`run_fidelity_gate`) per phase by default
- phases with no verification steps (e.g., documentation-only phases) must still satisfy a gate — the spec author can declare `gate_type = "manual_review"` or similar, but cannot declare zero required gates
- allow spec-level expansion to additional required gates, but not removal of minimum gate type

Controlled break-glass:
- optional privileged `gate-waiver` path with reason code and audit record
- waiver disabled by default (`allow_gate_waiver = false`) and never available to `autonomy_runner`

Acceptance:
- orchestrator cannot complete any phase without satisfying required gates
- orchestrator cannot complete spec if any phase is missing required gate completion
- any waiver is explicit, role-restricted, and observable

### P1: Proof-Carrying Progress (2-3 sprints)

### P1.1 One-time step proof token

Extend step model:
- add `step_proof` to `NextStep` / `LastStepIssued`
- require same token in `last_step_result`
- consume token exactly once

Implementation touchpoints:
- `src/foundry_mcp/core/autonomy/models.py`
- `src/foundry_mcp/core/autonomy/orchestrator.py`
- `src/foundry_mcp/core/autonomy/memory.py`
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py`

Idempotency:
- network retries are expected; a client may submit the same `(step_proof, outcome, payload)` tuple more than once if it didn't receive the server response
- the server MUST accept re-submission of an identical `(step_proof, payload_hash)` pair within a short grace window (configurable, default 30s) and return the same response — this is safe because the outcome is identical
- a re-submission with the same `step_proof` but a **different** payload is rejected as `PROOF_CONFLICT`
- after the grace window or after a new step is issued, the proof is fully consumed and any re-submission is rejected as `PROOF_EXPIRED`

Durability and atomicity (single-host requirement):
- persist proof state in session storage as an atomic record: `step_proof`, `payload_hash`, `consumed_at`, `grace_expires_at`, `response_hash`, and cached response payload
- consume proof and persist resulting response under one critical section guarded by per-session file lock
- replay acceptance after process restart must use persisted proof record (not in-memory cache)
- if persistence fails after proof validation, transition is failed and retriable with same payload (no partially consumed proof state)

Known limitation:
- the grace window means an observer who sees a `step_proof` in transit has a 30-second window to submit a different payload and receive a `PROOF_CONFLICT` error, confirming the proof was valid. This is a timing side channel. In the single-host stdio model this is low-risk (the observer would need process-level access, which is out of scope). Document this so it's not overlooked if the design is later adapted for multi-host use.

Acceptance:
- replay of old `last_step_result` fails deterministically
- mismatched or missing proof yields validation error
- identical re-submission within grace window returns the original response (idempotent)
- same proof with different payload rejected as `PROOF_CONFLICT`
- concurrent submissions for the same proof yield exactly one state transition

### P1.2 Proof-carrying verification success

For `execute_verification`:
- introduce server-issued verification receipt (hash of command/result metadata)
- `outcome="success"` requires valid receipt in `last_step_result`

Implementation options:
- extend verification tool output with receipt
- validate receipt in orchestrator before marking verification task complete

Acceptance:
- success cannot be claimed without receipt
- missing or invalid receipt yields deterministic validation error with recovery guidance

### P1.3 Gate evidence signature (single-host mode)

Upgrade `pending_gate_evidence`:
- include integrity checksum keyed by server secret
- verify checksum during `run_fidelity_gate` outcome consumption

Note: This is still same-host trust, but prevents accidental/malicious tampering through malformed client payloads.

Server secret provisioning:
- generated automatically on first server start if not present (random 32-byte key)
- stored in `$FOUNDRY_DATA_DIR/.server_secret` (outside repo tree, mode 0600)
- can be overridden via `FOUNDRY_MCP_GATE_SECRET` env var for deterministic testing
- on rotation: in-flight gate evidence signed with the old key becomes invalid — the orchestrator must re-request the gate step. This is acceptable because gate evidence is short-lived (scoped to the current step) and rotation is an operator-initiated action, not an automatic event.

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

### P2: Stronger Tamper Resistance + Operations (3+ sprints)

### P2.1 Append-only autonomy event ledger

**Scope clarification:** On a single-host deployment, "append-only" is a tamper **detection** guarantee, not tamper **prevention**. Any process with filesystem write access can truncate or modify the log. The value is post-hoc auditability and making tampering visible — not cryptographic immutability.

Add append-only log for state transitions:
- include hash chain (`prev_hash`, `event_hash`)
- events: step issued, step consumed, pause/resume, bypass/override, reset/end
- each entry includes timestamp, configured server role, server instance ID, action, and payload digest

Store under:
- `specs/.autonomy/audit/` (or configurable secure path)

Integrity verification:
- provide `foundry audit verify` CLI command that walks the hash chain and reports breaks
- the CLI and the programmatic check called on `session-start` must share a single implementation — no separate code paths
- verification runs automatically on `session-start` for existing audit trails
- broken chains produce a warning (not a hard block) — the log is forensic, not a gate

Acceptance:
- post-hoc integrity verification tool detects tampering (hash chain break, missing entries)
- verification tool reports the exact point of divergence

### P2.2 Runtime isolation profile for orchestrating agents

**Scope clarification:** This is application-level path and capability restriction, not OS-level sandboxing (seccomp/AppArmor/chroot). OS-level isolation is valuable but is a deployment concern outside this plan's scope.

Application-level hardening:
- `autonomy_runner` role is configured with an explicit `workspace_root` — all file operations are validated against this path prefix
- tool permission filtering: the runner's MCP session only exposes the tools in its role allowlist (P0.3), enforced at session initialization, not just at dispatch
- shell execution (`execute_verification`, etc.) runs with an environment that restricts `PATH` and kills subprocesses that block on stdin for longer than a configurable timeout (default 30s) — this is more robust than a command denylist
- deny `..` path traversal in all file path arguments for runner-scoped requests

Acceptance:
- runner cannot read/write outside its configured workspace root
- runner's MCP tool list excludes privileged actions at session level
- path traversal attempts rejected with explicit error
- interactive/blocking subprocesses are terminated after timeout

## Config and Contract Changes

### New Config (proposed)

All hardening flags ship with enforced defaults. No profile system — one set of secure defaults with explicit per-flag overrides for operators who need them:

```toml
[autonomy]
role = "observer"  # "autonomy_runner" | "maintainer" | "observer"

[autonomy.security]
allow_lock_bypass = false
require_reason_code_for_override = true
restrict_session_end_reset_to_privileged = true
enforce_required_phase_gates = true
allow_gate_waiver = false

[autonomy.rate_limit]
max_consecutive_denials = 10
denial_window_seconds = 60
retry_after_seconds = 5
```

Step proofs (P1.1) and verification receipts (P1.2) are always enforced once shipped — no toggle. Any explicit per-flag override takes precedence over the defaults.

Role allowlists:

```toml
[autonomy.roles.autonomy_runner]
allowed_actions = [
    "session-start", "session-resume", "session-step-next",
    "session-step-report", "session-heartbeat", "session-rebase",
    "run_fidelity_gate",
]
workspace_root = "/path/to/workspace"  # P2.2

[autonomy.roles.maintainer]
allowed_actions = ["*"]

[autonomy.roles.observer]
allowed_actions = [
    "list", "get", "view", "status", "search", "progress",
]
```

Environment variable override: `FOUNDRY_MCP_ROLE` sets `autonomy.role` (env takes precedence over config file).

### Response Contract Updates (proposed)

1. `next_step` includes `step_proof`.
2. override responses include structured `override_policy` metadata.
3. authorization failures return explicit `error_type="authorization"` with configured role, denied action, and required role in response body.
4. phase state includes required-gate status metadata (`required_phase_gates`, `satisfied_gates`, `missing_required_gates`).
5. blocked transitions return machine-readable gate-block details (phase id, gate type, blocking reason).
6. all blocking errors include `recovery_action` field with the specific action/parameters needed to unblock (see Recovery Protocol Requirements).
7. rate-limited responses include `retry_after` field.

## Test Plan

### Unit

1. Session/session-step fail closed when autonomy feature is disabled.
2. Role-based allowlist enforcement at dispatcher (runner denied mutation, maintainer allowed, observer read-only).
3. Bypass denied when config disallows it.
4. Step proof mismatch/replay rejection paths.
5. Step proof idempotent re-submission within grace window returns same response.
6. Step proof re-submission with different payload rejected as `PROOF_CONFLICT`.
7. Verification receipt required for success.
8. Phase-close/spec-complete blocked when required phase gates are unsatisfied.
9. Gate waiver denied for non-privileged roles and denied globally when `allow_gate_waiver=false`.
10. All blocking errors include `recovery_action` field.
11. Rate limiting triggers after N consecutive denials and returns `RATE_LIMITED` with `retry_after`.
12. Server started with no explicit role defaults to `observer`.

### Integration

1. Autonomy runner can complete happy path without privileged actions.
2. Runner cannot mutate task/lifecycle directly.
3. Maintainer can execute controlled override with reason code.
4. Reset/end denied to runner role.
5. Runner cannot advance across any phase boundary without completed required gate evidence.
6. Spec completion is denied if any phase has unmet gate requirements.
7. Two concurrent submissions of identical `last_step_result` for the same `step_proof` produce one transition and one replayed response.
8. Two concurrent submissions with same `step_proof` but different payloads deterministically return one success and one `PROOF_CONFLICT`.
9. Step-proof replay behavior remains correct after process restart.
### Property/Fuzz

1. Randomized step order / repeated step submissions cannot advance state.
2. Random payload tampering on gate evidence/step proof always rejected.
3. Randomized attempts to skip phase gates cannot produce phase-complete/spec-complete transitions.
4. Parallel randomized submissions for the same session cannot violate monotonic state version or proof single-consumption rules.

## Rollout Strategy

Each phase ships enforced — no feature flags, no compatibility windows.

1. Ship P0: all hardening active immediately (`allow_lock_bypass=false`, role enforcement active, phase gates required). Legacy specs that lack phase gate declarations hard-fail with remediation guidance via `foundry spec migrate-gates`.
2. Ship P1: step proofs and verification receipts always enforced. No proof-optional mode.
3. Ship P2: audit ledger and runtime isolation layered on top of the P0/P1 foundation.

### Preflight requirements

Before enabling hard-fail migration gates:
- run `foundry spec gates-audit` (new) to enumerate specs that will fail under required-gate invariants
- run `foundry spec migrate-gates --dry-run` to produce explicit patch previews
- block hardening enablement in production until audit returns zero blocking specs


## Success Metrics

1. `task.autonomy_write_lock.bypass.total` trends to near-zero.
2. Unauthorized mutation attempts blocked and observable via `authz.denied` metrics.
3. Zero successful state transitions from stale/replayed step submissions.
4. Zero verification success claims without valid receipt.
5. Incident recovery still possible for maintainer-role operators with auditable overrides.
6. Zero successful phase-complete/spec-complete transitions with unsatisfied required phase gates.
7. `authz.rate_limited` events observable but trending down (indicates confused agents are being caught and self-correcting).

## Open Decisions

1. ~~Principal identity source~~ — **Resolved:** Process-level role set at startup via config (`autonomy.role`) or environment variable (`FOUNDRY_MCP_ROLE`). The server runs over stdio transport where there is no per-request credential exchange; the role is immutable for the server's lifetime. Per-request authentication (API keys, OAuth, JWT) is deferred to a future plan revision if HTTP transport, multi-tenant, or remote executor support is added.

2. Step proof format:
- opaque random nonce (simpler, sufficient for single-host) vs HMAC payload token (stronger, needed if proofs cross trust boundaries).
- **Recommendation:** Start with opaque nonce for P1.1. Upgrade to HMAC if/when remote executor support is added.

3. Receipt granularity:
- command-level receipt only vs richer artifact checks (stdout hash, exit code, test IDs).
- **Recommendation:** Start with command-level receipt (command + exit code + output digest). Richer artifact checks can be layered on without contract changes.

4. ~~Backward-compat timeline~~ — **Resolved:** No compatibility windows. Each phase ships enforced.

5. Legacy spec migration behavior:
- auto-insert missing phase gates vs hard-fail with remediation guidance.
- **Recommendation:** Hard-fail with remediation guidance (provide `foundry spec migrate-gates` CLI command), but only after mandatory preflight audit/dry-run (`foundry spec gates-audit`, `foundry spec migrate-gates --dry-run`) confirms no unmanaged blockers.

6. ~~Emergency rollback path for hardening~~ — **Resolved:** No rollback path. Controls ship enforced once enabled.

## Recommended Execution Order

Ordered by dependency graph — each item's prerequisites are satisfied by prior items:

1. **P0.1** fail-closed gating for disabled autonomy (no dependencies)
2. **P0.2** bypass lockdown (independent, low-risk hardening)
3. **P0.3** role model + allowlist enforcement (ships enforced)
4. **P0.4** privileged override policy (requires P0.3 for role checks)
5. **P0.5** required phase-gate invariants (independent of authz, but benefits from P0.4 for waiver path)
6. **P1.1** step proof tokens
7. **P1.2** verification receipts
8. **P1.3** gate evidence signature
9. **P1.4** independent gate-audit checker
10. **P2.1** audit ledger (independent)
11. **P2.2** runtime isolation profiles (requires P0.3 for runner role scoping)

## Implementation Tickets

### Ticket HB-01: Fail-closed feature gating for autonomy handlers (P0.1)

Scope:
- Add explicit `autonomy_sessions` feature-flag checks at all 13 session/session-step handler entrypoints.
- The helper `_feature_disabled_response()` already exists in `handlers_session.py` but is never called.

Files:
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py` (modify — 9 handlers)
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py` (modify — 4 handlers)
- `src/foundry_mcp/config.py` (modify — ensure `feature_flags` dict is a proper config attribute, not just a test mock artifact)
- `tests/unit/test_core/autonomy/test_handlers_session.py` (modify)
- `tests/unit/test_core/autonomy/test_handlers_session_step.py` (modify)

Implementation:
- Add a shared guard function (or inline check) at the top of each handler that reads `config.feature_flags.get("autonomy_sessions", False)` and returns `_feature_disabled_response(action, request_id)` when disabled.
- Ensure `feature_flags` is a first-class `ServerConfig` field with a default of `{}`, loaded from config/env. Currently tests set it as a bare attribute on mock config objects — make this real.
- The `_feature_disabled_response` helper in `handlers_session_step.py` does not exist yet — add one or import from `handlers_session.py` via a shared location.

Handlers to gate (exhaustive list):
- `handlers_session.py`: `session-start`, `session-pause`, `session-resume`, `session-end`, `session-status`, `session-list`, `session-rebase`, `session-heartbeat`, `session-reset`
- `handlers_session_step.py`: `session-step-next`, `session-step-report`, `session-step-replay`, `session-step-heartbeat`

Acceptance tests:
- `pytest tests/unit/test_core/autonomy/test_handlers_session.py -k "feature_disabled"`
- `pytest tests/unit/test_core/autonomy/test_handlers_session_step.py -k "feature_disabled"`

Done when:
- Every session and session-step handler returns `FEATURE_DISABLED` when `autonomy_sessions` is off.
- No autonomy state is created or mutated when the feature is disabled.

### Ticket HB-02: Restrict write-lock bypass by default (P0.2)

Scope:
- Add config toggle `autonomy.security.allow_lock_bypass` (default `false`).
- When disabled, reject `bypass_autonomy_lock=true` in all mutation handlers regardless of caller input.

Files:
- `src/foundry_mcp/config.py` (modify — add `allow_lock_bypass` field)
- `src/foundry_mcp/core/autonomy/write_lock.py` (modify — `check_autonomy_write_lock()`)
- `src/foundry_mcp/tools/unified/task_handlers/_helpers.py` (modify — `_check_autonomy_write_lock()` wrapper)
- `tests/unit/test_core/autonomy/test_write_lock.py` (modify)

Implementation:
- Add `allow_lock_bypass: bool = False` to the autonomy security config section.
- In `check_autonomy_write_lock()`: before evaluating `bypass_flag`, check the config toggle. If `allow_lock_bypass` is `false` and `bypass_flag` is `true`, return `WriteLockStatus.LOCKED` with an error message indicating bypass is globally disabled. Emit a `write_lock.bypass_denied` metric.
- The existing `bypass_reason` validation (reject empty reason) remains — it's a second layer of defense when bypass is enabled.
- No changes needed in the 15+ individual mutation handlers (`handlers_mutation.py`, `handlers_batch.py`, `handlers_lifecycle.py`) — they already pass `bypass_autonomy_lock` through to the shared check function.

Acceptance tests:
- `pytest tests/unit/test_core/autonomy/test_write_lock.py -k "bypass_denied_by_config"`
- `pytest tests/unit/test_core/autonomy/test_write_lock.py -k "bypass_allowed_when_config_permits"`

Done when:
- `bypass_autonomy_lock=true` is rejected by default across all mutation routes.
- Bypass only works when `allow_lock_bypass=true` is explicitly set in config.
- Denied bypass attempts are metricated.

### Ticket HB-03: Authorization module + process-level role (P0.3a)

Scope:
- Create the authorization module with role resolution and action allowlist checking.
- Add `autonomy.role` to config with env var override.
- Wire role into request context so it's available throughout the call chain.

Files:
- `src/foundry_mcp/core/authorization.py` (create)
- `src/foundry_mcp/config.py` (modify — add `role` field to autonomy config, add role allowlist config)
- `src/foundry_mcp/core/context.py` (modify — add `server_role` ContextVar and accessor)
- `src/foundry_mcp/server.py` (modify — initialize role at startup, set context var)
- `tests/unit/test_core/test_authorization.py` (create)

Implementation:
- `authorization.py` exports:
  - `check_action_allowed(role: str, tool_name: str, action: str) -> AuthzResult` — checks `(role, action)` against the allowlist. Returns an `AuthzResult` dataclass with `allowed: bool`, and on denial: `denied_action`, `configured_role`, `required_role`.
  - `get_role_allowlist(role: str) -> set[str]` — returns the allowed actions for a role.
  - Role allowlists loaded once from config at import/init time.
- Config additions:
  - `autonomy.role` (default `"observer"`), overridden by `FOUNDRY_MCP_ROLE` env var.
  - `autonomy.roles.*` sections with `allowed_actions` lists (use the defaults from the plan if not configured).
- Context additions:
  - `server_role_var: ContextVar[str]` with default `"observer"`.
  - `get_server_role() -> str` accessor.
  - `sync_request_context()` and `async_request_context()` accept optional `server_role` param.
- Server init:
  - Read role from config, set `server_role_var` at process start.
  - Log the configured role at startup.

Acceptance tests:
- `pytest tests/unit/test_core/test_authorization.py -k "runner_allowed_session_actions"`
- `pytest tests/unit/test_core/test_authorization.py -k "runner_denied_mutation_actions"`
- `pytest tests/unit/test_core/test_authorization.py -k "maintainer_wildcard"`
- `pytest tests/unit/test_core/test_authorization.py -k "observer_read_only"`
- `pytest tests/unit/test_core/test_authorization.py -k "unconfigured_role_defaults_observer"`
- `pytest tests/unit/test_core/test_authorization.py -k "env_var_overrides_config"`

Done when:
- `check_action_allowed()` correctly enforces `autonomy_runner`, `maintainer`, and `observer` allowlists.
- Server role is readable from context anywhere in the call chain via `get_server_role()`.
- `FOUNDRY_MCP_ROLE` env var overrides the config file value.
- Unconfigured role defaults to `observer`.

### Ticket HB-04: Integrate authorization into dispatch chain (P0.3b)

Scope:
- Wire `check_action_allowed()` into the unified dispatch flow so every action is checked against the server's role before handler execution.
- Return proper `AUTHORIZATION` error responses with recovery guidance.

Files:
- `src/foundry_mcp/tools/unified/common.py` (modify — `dispatch_with_standard_errors()`)
- `src/foundry_mcp/core/responses.py` (modify — add `AUTHORIZATION` error code if not present)
- `tests/tools/unified/test_dispatch_common.py` (modify — add auth enforcement tests across all routers)
- `tests/tools/unified/test_common.py` (modify — unit tests for auth in dispatch)

Implementation:
- In `dispatch_with_standard_errors()`, after action validation and before `router.dispatch()`:
  - Call `check_action_allowed(get_server_role(), tool_name, action)`.
  - On denial, return an `AUTHORIZATION` error response with `role`, `action`, and `required_role` in details, plus `recovery_action` guidance.
- Error precedence order (already specified in plan): `FEATURE_DISABLED` → action validation → `AUTHORIZATION` → argument validation. The feature-flag check (P0.1) happens inside handlers, so auth naturally falls between the router's action validation and handler execution.
- Add parametrized tests across all 14 routers confirming that a restricted role cannot call privileged actions.

Acceptance tests:
- `pytest tests/tools/unified/test_dispatch_common.py -k "authorization_denied"`
- `pytest tests/tools/unified/test_dispatch_common.py -k "authorization_allowed"`
- `pytest tests/tools/unified/test_common.py -k "authorization_error_response"`

Done when:
- Every tool's dispatch path checks role authorization before invoking the handler.
- Denied requests return `AUTHORIZATION` error with role, action, and recovery guidance.
- `authz.denied` metrics are emitted on denial.

### Ticket HB-05: Rate limiting on authorization denials (P0.3c)

Scope:
- Add in-memory rate limiting that triggers after repeated consecutive authorization denials.
- Prevents confused agents from hammering denied endpoints.

Files:
- `src/foundry_mcp/core/authorization.py` (modify — add rate limit tracker)
- `src/foundry_mcp/tools/unified/common.py` (modify — check rate limit before dispatch)
- `src/foundry_mcp/config.py` (modify — add `autonomy.rate_limit` config section)
- `tests/unit/test_core/test_authorization.py` (modify)

Implementation:
- Add `RateLimitTracker` class to `authorization.py`:
  - Tracks consecutive denials per action within a sliding window.
  - `check_rate_limit(action: str) -> Optional[float]` — returns `retry_after` seconds if limited, `None` if allowed.
  - `record_denial(action: str)` — increments counter.
  - `reset(action: str)` — called on successful dispatch to clear the counter.
  - State is in-memory only — resets on process restart, which is acceptable.
- Config: `max_consecutive_denials` (default 10), `denial_window_seconds` (default 60), `retry_after_seconds` (default 5).
- In `dispatch_with_standard_errors()`: check rate limit before auth check. If limited, return `RATE_LIMITED` error with `retry_after` field. If auth check denies, call `record_denial()`. If dispatch succeeds, call `reset()`.

Acceptance tests:
- `pytest tests/unit/test_core/test_authorization.py -k "rate_limit_triggers_after_threshold"`
- `pytest tests/unit/test_core/test_authorization.py -k "rate_limit_resets_on_success"`
- `pytest tests/unit/test_core/test_authorization.py -k "rate_limit_window_expires"`
- `pytest tests/unit/test_core/test_authorization.py -k "rate_limit_response_includes_retry_after"`

Done when:
- After N consecutive denials within the window, the server returns `RATE_LIMITED` instead of `AUTHORIZATION`.
- Rate limit resets on successful dispatch or after the window expires.
- `authz.rate_limited` metric is emitted.

### Ticket HB-06: Harden escape hatch policy (P0.4)

Scope:
- Restrict `session-end` and `session-reset` to `maintainer` role.
- Add mandatory structured reason code (closed enum) for privileged overrides.
- Extend lock bypass paths to require `maintainer` role when bypass is enabled.

Files:
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py` (modify — `_handle_session_end`, `_handle_session_reset`)
- `src/foundry_mcp/core/autonomy/models.py` (modify — add `OverrideReasonCode` enum)
- `src/foundry_mcp/core/autonomy/write_lock.py` (modify — role check on bypass path)
- `tests/unit/test_core/autonomy/test_handlers_session.py` (modify)
- `tests/unit/test_core/autonomy/test_write_lock.py` (modify)

Implementation:
- `session-end` and `session-reset` are already restricted at the dispatch level by the role allowlist (HB-03/HB-04) — they're not in `autonomy_runner`'s allowed actions. This ticket adds the **reason code** requirement as a second layer of defense inside the handlers.
- Add `OverrideReasonCode` enum to models: `STUCK_AGENT`, `CORRUPT_STATE`, `OPERATOR_OVERRIDE`, `INCIDENT_RESPONSE`, `TESTING`. Free-text `reason_detail` is optional but the enum code is mandatory.
- In `_handle_session_end` and `_handle_session_reset`: validate that `reason_code` is present and is a valid enum member. Return `VALIDATION_ERROR` if missing or invalid.
- In `check_autonomy_write_lock()`: when bypass is allowed by config and `bypass_flag=true`, additionally require that the server role is `maintainer`. If role is `autonomy_runner`, reject bypass even when config allows it.
- Audit: journal entries for end/reset/bypass include the reason code and configured role.

Acceptance tests:
- `pytest tests/unit/test_core/autonomy/test_handlers_session.py -k "session_end and reason_code"`
- `pytest tests/unit/test_core/autonomy/test_handlers_session.py -k "session_reset and reason_code"`
- `pytest tests/unit/test_core/autonomy/test_handlers_session.py -k "session_end_missing_reason_rejected"`
- `pytest tests/unit/test_core/autonomy/test_write_lock.py -k "bypass_requires_maintainer_role"`

Done when:
- `session-end` and `session-reset` require a valid `OverrideReasonCode`.
- Lock bypass requires `maintainer` role even when `allow_lock_bypass=true`.
- All privileged override actions include reason code in journal entries.

### Ticket HB-07: Add required-gate state model + migration (P0.5a)

Scope:
- Introduce explicit required gate tracking fields in autonomy state models.
- Bump state schema and add migration for existing session files.

Files:
- `src/foundry_mcp/core/autonomy/models.py` (modify)
- `src/foundry_mcp/core/autonomy/state_migrations.py` (modify)
- `tests/unit/test_core/autonomy/test_context_tracker.py` (modify)
- `tests/unit/test_core/autonomy/test_memory.py` (modify)
- `tests/unit/test_core/autonomy/conftest.py` (modify)

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

### Ticket HB-08: Compute required phase gates at session start/rebase (P0.5b)

Scope:
- Compute and persist `required_phase_gates` during `session-start` and `session-rebase`.
- Ensure spec changes reconcile required gate obligations without silently dropping required gates.

Files:
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py` (modify)
- `src/foundry_mcp/core/autonomy/models.py` (modify)
- `tests/unit/test_core/autonomy/test_handlers_session.py` (modify)

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

### Ticket HB-09: Enforce phase/spec completion invariants in orchestrator (P0.5c)

Scope:
- Block phase-complete and spec-complete transitions when required gates are unsatisfied.
- Return deterministic blocking errors with phase/gate metadata and recovery guidance.

Files:
- `src/foundry_mcp/core/autonomy/orchestrator.py` (modify)
- `src/foundry_mcp/core/autonomy/models.py` (modify)
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py` (modify)
- `tests/unit/test_core/autonomy/test_orchestrator.py` (modify)
- `tests/unit/test_core/autonomy/test_handlers_session_step.py` (modify)

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

### Ticket HB-10: Add privileged gate-waiver path (default off) (P0.5d)

Scope:
- Add controlled break-glass override for required-gate invariant failures.
- Restrict waiver to `maintainer` role and structured reason codes.

Files:
- `src/foundry_mcp/tools/unified/task_handlers/__init__.py` (modify)
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py` (modify)
- `src/foundry_mcp/core/autonomy/models.py` (modify)
- `src/foundry_mcp/core/authorization.py` (modify — created in HB-03)
- `tests/unit/test_core/autonomy/test_handlers_session.py` (modify)

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

### Ticket HB-11: Independent required-gate audit checker (P1.4)

Scope:
- Add server-side recomputation of required gate obligations and compare with recorded evidence before terminal transitions.

Files:
- `src/foundry_mcp/core/autonomy/orchestrator.py` (modify)
- `src/foundry_mcp/core/autonomy/models.py` (modify)
- `tests/unit/test_core/autonomy/test_orchestrator.py` (modify)

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

### Ticket HB-12: Contract and observability updates for gate invariants (P0.5e)

Scope:
- Expose required-gate status and gate-block errors in API contract and telemetry.
- Ensure discoverability and rollout controls are explicit.

Files:
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py` (modify)
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py` (modify)
- `src/foundry_mcp/core/discovery.py` (modify)
- `src/foundry_mcp/config.py` (modify)
- `tests/unit/test_core/autonomy/test_handlers_session_step.py` (modify)

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

### Ticket HB-13: End-to-end invariant enforcement test (P0.5f)

Scope:
- Add integration test to prove orchestrator cannot complete a spec when a required phase gate is skipped.

Files:
- `tests/unit/test_core/autonomy/test_integration.py` (modify — existing integration test file)

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

## Future Work

Items explicitly deferred from this plan that should be revisited when the deployment model changes:

1. **Dual-control overrides** — Requires two distinct authenticated identities confirming a destructive action. This is not feasible with process-level identity over stdio. Revisit when HTTP transport is added or when multi-process coordination is implemented. The architectural approach would be: first process requests override → server writes a time-limited approval token to a shared location → second process (different role) confirms with the token → both identities recorded in audit ledger.

2. **Per-request authentication** — API keys, OAuth, JWT. Required for multi-tenant deployments or when the server is exposed over HTTP. The authorization module (P0.3) is designed so the role resolution can be swapped from "read config at startup" to "extract from request" without changing the enforcement layer.

3. **Cryptographic trust across remote executors** — HMAC-signed step proofs, remote attestation for verification results. Required when proof tokens cross host boundaries.
