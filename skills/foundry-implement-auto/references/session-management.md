# Session Management

## Contents

- [Session Start](#session-start)
- [Session Reuse](#session-reuse)
- [Stale Session Recovery](#stale-session-recovery)
- [Spec Drift and Rebase](#spec-drift-and-rebase)

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

## Session Reuse

If the server returns `SPEC_SESSION_EXISTS` on session start:

1. **List candidates.** List non-terminal sessions for the spec (limit 5).
2. **Require unambiguous candidate.** Exactly one candidate must exist. If ambiguous, stop with `SESSION_REUSE_AMBIGUOUS`.
3. **Validate compatibility.** The existing session must have both:
   - `stop_on_phase_completion=true`
   - `write_lock_enforced=true`
   Both are required for hardened one-phase semantics.
4. **Reject incompatible sessions.** Stop with `SESSION_REUSE_INCOMPATIBLE` and require explicit operator intervention. The skill never force-ends or reconfigures an existing session.

### Decision Flow

```
SPEC_SESSION_EXISTS
  → List sessions (limit 5)
    → [count != 1?] → SESSION_REUSE_AMBIGUOUS → EXIT
    → [count == 1]
      → [stop_on_phase_completion && write_lock_enforced?]
        → [yes] → Reuse session
        → [no] → SESSION_REUSE_INCOMPATIBLE → EXIT
```

## Stale Session Recovery

When a previous agent died mid-session (e.g., error threshold pause followed by agent exit), a new agent inherits a session with an unreported pending step. The step proof from that step is bound to the prior agent context and lost. Calling `session-step next` without the proof fails with `STEP_PROOF_MISSING`.

Replay-based recovery resolves this without counter resets, session cycling, or gate bypass.

### Recovery Sequence

Stale session recovery runs **only on session reuse** (after `SPEC_SESSION_EXISTS` → reuse validation passes). Fresh sessions cannot have stale steps, so this probe is skipped for new sessions.

After session reuse validation passes, before entering the step loop:

1. **Probe for pending step.** Call `task(action="session-step", command="replay", session_id=...)`.
2. **Branch on result:**

```
session-step replay
  → [Transport error / malformed response]
    → Treat as blocked_runtime → EXIT escalation. Do not retry or fall through to session-step next.
  → [NOT_FOUND]
    → No unreported step. Enter step loop normally with session-step next.
  → [Success: cached response]
    → Read loop_signal from response:
      → [paused_needs_attention | failed | blocked_runtime]
        → EXIT escalation (deterministic exit table applies)
      → [continue]
        → Validate next_step is present in response (if absent, treat as blocked_runtime → EXIT escalation)
        → Extract next_step + step_proof from cached response
        → Dispatch step through normal step handlers
        → Report outcome (report response provides a fresh step_proof for the next iteration)
        → Enter step loop normally
      → [phase_complete | spec_complete]
        → EXIT success
```

### Why Replay Is Safe

| Property | Guarantee |
|----------|-----------|
| **Read-only** | Returns cached `last_issued_response`. Does not re-execute, mutate state, or advance the step pointer. |
| **No counter reset** | Error counters, fidelity cycle counters, and task counters are untouched. A session paused at `error_threshold` stays paused. |
| **No gate bypass** | Phase gate status unchanged. The recovered step still requires its original proof token for the report. |
| **Idempotent** | Multiple replay calls return the same cached response. Session state is identical before and after. |

### State-Specific Behavior

| Replayed State | Agent Action | Rationale |
|----------------|-------------|-----------|
| `loop_signal: paused_needs_attention` | EXIT escalation | Prior agent was correctly paused. New agent must not un-pause — requires operator. |
| `loop_signal: failed` | EXIT escalation | Session failed. Operator must investigate. |
| `loop_signal: blocked_runtime` | EXIT escalation | Runtime policy violation. Cannot self-recover. |
| `loop_signal: continue` with `next_step` | Dispatch the step | Prior agent's last issued step, never reported. New agent has the proof from cached response and can execute and report normally. |
| `loop_signal: continue` without `next_step` | EXIT escalation (`blocked_runtime`) | Defensive: cached response is malformed or session state is inconsistent. Escalate rather than guessing. |
| `loop_signal: phase_complete` | EXIT success | Phase already completed. |
| `loop_signal: spec_complete` | EXIT success | Spec already completed. |
| `NOT_FOUND` (no cached response) | Call `session-step next` | All prior steps were reported. Session is clean. |
| Transport error / malformed response | EXIT escalation (`blocked_runtime`) | Replay call itself failed. Do not retry in a loop or fall through to `next`. Escalate for operator investigation. |

### Edge Cases

1. **Proof expired.** The proof is in the cached response but the server-side grace window may have elapsed. The report returns `STEP_PROOF_EXPIRED` → `blocked_runtime` → agent escalates correctly.
2. **Concurrent agents.** Replay is read-only and idempotent. Only the first agent to report advances the session. Others get `STEP_PROOF_CONFLICT` → escalate.
3. **Cached response is a pause step.** `next_step.type == pause` → step handler exits with escalation. No special logic needed.
4. **Fresh proof on report.** After the agent dispatches the recovered step and reports its outcome, the report response issues a fresh `step_proof` for the next `session-step next` call. The stale proof from the cached response is consumed on report and not reused.
5. **Replay transport failure.** If the replay MCP call itself errors (network failure, server 500, malformed response envelope), treat as `blocked_runtime` and escalate immediately. Do not retry replay in a loop or silently fall through to `session-step next` — the session state is unknown.

## Spec Drift and Rebase

If the spec is modified mid-session:
- Surface `session-rebase` guidance to the operator
- The skill cannot rebase — only `maintainer` role can
- Do not attempt to work around drift; escalate immediately
