# Session Management

## Contents

- [Session Start](#session-start)
- [Session Reuse](#session-reuse)
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

## Spec Drift and Rebase

If the spec is modified mid-session:
- Surface `session-rebase` guidance to the operator
- The skill cannot rebase — only `maintainer` role can
- Do not attempt to work around drift; escalate immediately
