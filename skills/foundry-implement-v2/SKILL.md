---
name: foundry-implement-v2
description: Executes one autonomous implementation phase for a spec using Foundry MCP session-step orchestration under unattended posture with autonomy_runner role, deterministic loop-signal exits, and verification receipt construction.
---

# foundry-implement-v2

## Purpose

Run exactly one autonomous implementation phase for a spec in unattended mode, then stop deterministically at the phase boundary or any escalation condition. Designed for maximal hardening compatibility — this skill never requests elevated privileges, never bypasses safety controls, and treats any relaxation of server policy as an escalation condition.

Reference:
- Runtime contract: `src/foundry_mcp/skills/foundry_implement_v2.py`
- Supervisor runbook: `docs/guides/autonomy-supervisor-runbook.md`
- Configuration guide: `docs/06-configuration.md`
- Sample config: `samples/foundry-mcp.toml`

## Environment Prerequisites

The server must be configured for unattended autonomous operation before invocation. The expected environment:

```sh
# Required: role and posture
FOUNDRY_MCP_ROLE=autonomy_runner
FOUNDRY_MCP_AUTONOMY_POSTURE=unattended

# Required: feature flags
FOUNDRY_MCP_FEATURE_FLAG_AUTONOMY_SESSIONS=true
FOUNDRY_MCP_FEATURE_FLAG_AUTONOMY_FIDELITY_GATES=true
```

The `unattended` posture applies the following server-side defaults automatically. The skill does not set these — it verifies them:

| Setting | Unattended default | Purpose |
|---|---|---|
| `role` | `autonomy_runner` | Restricted allowlist (14 actions) |
| `allow_lock_bypass` | `false` | Write lock cannot be bypassed |
| `allow_gate_waiver` | `false` | Gates cannot be waived |
| `enforce_required_phase_gates` | `true` | Phase transitions blocked without gate satisfaction |
| `gate_policy` | `strict` | Gate failures block progression |
| `stop_on_phase_completion` | `true` | One-phase primitive enforced |
| `auto_retry_fidelity_gate` | `true` | Automatic feedback → gate retry cycle |
| `max_consecutive_errors` | `3` | Pause after 3 consecutive step failures |
| `max_fidelity_review_cycles_per_phase` | `3` | Prevent infinite gate retry spinning |

Equivalent TOML (alternative to env vars):

```toml
[feature_flags]
autonomy_sessions = true
autonomy_fidelity_gates = true

[autonomy_posture]
profile = "unattended"
```

## Invocation Contract

```sh
claude -p /foundry-implement-v2 <spec-id> .
```

- Arg 1: `spec_id` — the spec to execute.
- Arg 2: workspace path (default `.`).

The skill verifies all MCP-level prerequisites at startup and fails fast with remediation guidance if anything is misconfigured. For agent-level isolation requirements (filesystem, git, config protection), see `docs/guides/autonomy-agent-isolation.md`.

## Non-Negotiable Rules

1. **Runtime truth only.** Use MCP tool responses as the source of truth for feature flags, posture, and role. Never trust discovery metadata or manifest alone.
2. **Server-driven sequencing.** The session-step orchestrator is the sole authority on task ordering, phase boundaries, and gate timing. Never infer or override task sequence.
3. **No privilege escalation.** Never set `allow_lock_bypass=true`, `allow_gate_waiver=true`, or request elevated role. Always pass `enforce_autonomy_write_lock=true` on session start.
4. **Fail fast on policy violations.** Any preflight failure (role denied, feature disabled, wrong posture, spec not found) terminates immediately with structured remediation.
5. **Deterministic exit.** Stop on `phase_complete` or `spec_complete`. Escalate on all other `loop_signal` values. Never continue on ambiguous state.
6. **Bounded execution.** Enforce a hard loop iteration limit (`max_iterations=200`). If exceeded without terminal signal, stop and escalate — never spin indefinitely.

## Startup Preflight

All six checks must pass before session start. Any failure produces a structured error (`FoundryImplementV2Error`) with error code and remediation hint.

1. **Resolve spec.**
   - `spec(action="find", spec_id=...)`.
   - Fail: `SPEC_RESOLUTION_FAILED`.

2. **Detect action shape.**
   - Probe canonical shape: `task(action="session", command="list", limit=1)`.
   - Fallback to legacy shape: `task(action="session-list", limit=1)`.
   - Non-terminal errors (`AUTHORIZATION`, `FEATURE_DISABLED`) do not fail the probe — they indicate the shape is accepted but gated by policy.
   - Fail: `ACTION_SHAPE_UNSUPPORTED`.

3. **Verify feature flags.**
   - `server(action="capabilities")` → read `data.runtime.autonomy.enabled_now`.
   - Require `autonomy_sessions == true`.
   - Require `autonomy_fidelity_gates == true` (when fidelity-gated execution is expected, which is the default).
   - Fail: `FEATURE_DISABLED`.

4. **Verify posture profile.**
   - Read `data.runtime.autonomy.posture_profile` from the capabilities response.
   - Reject `debug` — it disables gate enforcement and allows uncontrolled escape hatches, making it incompatible with unattended operation.
   - Accept `unattended` (preferred) or `supervised` (human-in-the-loop provides equivalent oversight).
   - Fail: `POSTURE_UNSUPPORTED`.

5. **Verify role authorization.**
   - Using the detected action shape, call session list with `limit=1`.
   - If the response returns `AUTHORIZATION` or `FEATURE_DISABLED`, the `autonomy_runner` role either isn't set or doesn't have session access.
   - Fail: `AUTHORIZATION` or `ROLE_PREFLIGHT_FAILED`.

6. **Separation principle.**
   - Action-shape detection (step 2) and role verification (step 5) are separate probes. Shape detection determines which action format to use. Role verification confirms the caller has permission. They must not be conflated.

## Session Start

Start a session with hardened one-phase defaults:

```json
{
  "action": "session",
  "command": "start",
  "spec_id": "...",
  "gate_policy": "strict",
  "stop_on_phase_completion": true,
  "auto_retry_fidelity_gate": true,
  "enforce_autonomy_write_lock": true,
  "idempotency_key": "..."
}
```

These parameters are intentionally hardcoded in the skill, not configurable. They must match or exceed the server's unattended posture defaults. The skill never sends `allow_lock_bypass`, `allow_gate_waiver`, or any parameter that would weaken server-side enforcement.

### Session Reuse

If the server returns `SPEC_SESSION_EXISTS`:

1. List non-terminal sessions for the spec (limit 5).
2. Require exactly one candidate. If ambiguous, stop (`SESSION_REUSE_AMBIGUOUS`).
3. Validate compatibility: the existing session must have both `stop_on_phase_completion=true` and `write_lock_enforced=true`. Both are required for hardened one-phase semantics.
4. If incompatible, stop (`SESSION_REUSE_INCOMPATIBLE`) and require explicit operator intervention. The skill never force-ends or reconfigures an existing session.

## Step Loop

Repeat (up to `max_iterations=200`):

1. Call `task(action="session-step", command="next", session_id=...)`.
2. Read `data.loop_signal` and apply deterministic exits:
   - `phase_complete` → success stop (single phase done).
   - `spec_complete` → success stop.
   - `paused_needs_attention` → escalation stop.
   - `failed` → escalation stop.
   - `blocked_runtime` → escalation stop.
3. If no terminal signal and response is successful, extract `data.next_step` and dispatch by `next_step.type`.
4. Report outcome via the appropriate transport:
   - **Simple reports** (no extended fields): `task(action="session-step", command="report", session_id=..., step_id=..., step_type=..., outcome=..., note=..., files_touched=...)`.
   - **Extended reports** (includes `verification_receipt`, `gate_attempt_id`, `step_proof`, `task_id`, or `phase_id`): `task(action="session-step", command="next", session_id=..., last_step_result={...})`.
   - Rationale: `session-step-report` is optimized for small payloads; extended fields require the full `last_step_result` envelope.

If `max_iterations` is exhausted without a terminal signal, stop with `LOOP_LIMIT_EXCEEDED`.

## Step Handlers

### `implement_task`

1. Read task details from `next_step.instruction`.
2. Use `task(action="prepare", spec_id=...)` as needed to understand scope.
3. Apply code changes.
4. Run relevant checks.
5. Report `last_step_result` with `task_id`, `note`, and `files_touched`.

### `execute_verification`

1. Run the verification command specified in `next_step.instruction`, or call `verification(action="execute", ...)`.
2. Construct a `verification_receipt` with all required fields. Under strict gate policy, the orchestrator validates every field:
   - `command_hash`: SHA-256 hex digest of the verification command string (lowercase, exactly 64 characters).
   - `exit_code`: integer exit code from the verification process.
   - `output_digest`: SHA-256 hex digest of the verification output (lowercase, exactly 64 characters).
   - `issued_at`: UTC timestamp with timezone info (ISO 8601). Naive datetimes are rejected by the server.
   - `step_id`: must match the current step's `step_id`. Mismatched binding is rejected.
3. Use the canonical helper `issue_verification_receipt(step_id=..., command=..., exit_code=..., output=...)` from `foundry_mcp.core.autonomy.models` to construct valid receipts. Do not hand-construct receipt objects — the helper enforces hash format, timezone, and binding invariants.
4. Include the receipt in `last_step_result.verification_receipt`.
5. If the receipt is missing or malformed, the server returns `ERROR_VERIFICATION_RECEIPT_MISSING` or `ERROR_VERIFICATION_RECEIPT_INVALID`, which maps to `blocked_runtime`.

### `run_fidelity_gate`

1. Call `review(action="fidelity-gate", spec_id=..., session_id=..., phase_id=..., step_id=...)`.
2. Capture `gate_attempt_id` from the response — this is required in the step report.
3. Report gate outcome in `last_step_result` with `gate_attempt_id` and `phase_id`.
4. Under strict gate policy, the server validates gate evidence integrity via checksum. A checksum failure returns `ERROR_GATE_INTEGRITY_CHECKSUM` (`blocked_runtime`).

### `address_fidelity_feedback`

1. Retrieve fidelity findings from `next_step.instruction`.
2. Remediate code and/or tests to address the findings.
3. Report changed files, outcome, and `phase_id`.
4. The orchestrator limits fidelity review cycles per phase (default 3). Exceeding the limit triggers `fidelity_cycle_limit` pause reason (`paused_needs_attention`).

### `pause` and `complete_spec`

- `pause`: stop the loop and return an escalation packet with the pause reason.
- `complete_spec`: success stop — the spec is fully implemented.

## Step Proof Handling

The orchestrator includes a `step_proof` token on emitted steps. This is a one-time-use token for replay protection and is enforced under unattended posture:

- If `next_step.step_proof` is present, include it unchanged in `last_step_result.step_proof`.
- The server enforces one-time consumption with a grace window for transport retries. Error codes:
  - `STEP_PROOF_MISSING`: proof required but not included in report.
  - `STEP_PROOF_MISMATCH`: proof does not match the expected token.
  - `STEP_PROOF_CONFLICT`: proof was already consumed (double-submit).
  - `STEP_PROOF_EXPIRED`: proof grace window has elapsed.
- All proof errors map to `blocked_runtime` loop signal and trigger immediate escalation.
- The skill passes proof tokens through verbatim. It does not generate, modify, or validate proofs — that is the server's responsibility.

## Deterministic Exit Table

| `loop_signal` | Exit type | `final_status` | Action |
|---|---|---|---|
| `phase_complete` | Success stop | `paused` | Single phase done. Supervisor may queue next phase. |
| `spec_complete` | Success stop | `completed` | Spec fully implemented. Close supervisor run. |
| `paused_needs_attention` | Escalation stop | `paused` | Route to operator (gate findings, limits, staleness, context threshold). |
| `failed` | Escalation stop | `failed` | Investigate failure before retry or rebase. |
| `blocked_runtime` | Escalation stop | varies | Resolve authorization, feature, integrity, or proof errors first. |

Error codes that map to `blocked_runtime` (escalation, not retry): `AUTHORIZATION`, `FEATURE_DISABLED`, `ERROR_REQUIRED_GATE_UNSATISFIED`, `ERROR_GATE_AUDIT_FAILURE`, `ERROR_GATE_INTEGRITY_CHECKSUM`, `ERROR_INVALID_GATE_EVIDENCE`.

## Error Handling

1. For retry-safe transport repeats (network failure, timeout), use `task(action="session-step", command="replay", session_id=...)`. The server returns the cached response for the most recent step without re-executing.
2. On spec drift (spec modified mid-session), surface `session-rebase` guidance to the operator. The skill cannot rebase — only `maintainer` role can.
3. Never mutate task state outside the session-step protocol. Never call `task(action="complete", ...)` or `task(action="update", ...)` directly — the orchestrator manages task state transitions.
4. All structured errors include `code`, `message`, and `remediation`. Key error code categories:
   - **Startup**: `SPEC_RESOLUTION_FAILED`, `ACTION_SHAPE_UNSUPPORTED`, `CAPABILITIES_UNAVAILABLE`, `FEATURE_DISABLED`, `POSTURE_UNSUPPORTED`, `AUTHORIZATION`.
   - **Session**: `SESSION_START_FAILED`, `SESSION_REUSE_AMBIGUOUS`, `SESSION_REUSE_INCOMPATIBLE`.
   - **Step dispatch**: `STEP_TYPE_UNSUPPORTED`, `GATE_ATTEMPT_REQUIRED`, `VERIFICATION_RECEIPT_REQUIRED`.
   - **Loop**: `STEP_MISSING`, `LOOP_LIMIT_EXCEEDED`.

## Exit Payload

Always emit a complete exit packet:

- `spec_id`
- `session_id`
- `final_status`
- `loop_signal`
- `pause_reason` (if paused)
- `active_phase_id`
- `last_step_id`
- `recommended_actions` (if present — machine-readable escalation actions from the server)
- `details.response_success` — whether the last server response was successful
- `details.error_code` (if applicable)
- `details.recommended_actions` (if applicable — structured list with `action`, `description`, `command`)
- concise summary

The exit packet is the skill's only output. The supervisor uses it to decide whether to queue the next phase, alert an operator, or terminate the run.
