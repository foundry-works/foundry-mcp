# Autonomous Posture Hardening

Date: 2026-02-16
Scope: Hardening posture for phase-looped autonomous execution using `session-*`/`session-step-*` + fidelity gates.

## Purpose

Define a concrete hardening posture so autonomous looping can run safely without relying on soft controls (prompt instructions) as the primary boundary.

## Security Objective

Prevent silent bypass of sequencing and quality controls while preserving controlled operator override when absolutely necessary.

## Threat Model (Practical)

Assume these failure modes are plausible:
1. The agent is confused and attempts unintended commands.
2. The operator accidentally grants too-broad authority.
3. The loop encounters repeated failures and starts thrashing.
4. State/evidence drift occurs across retries/resumes.
5. A privileged actor can attempt to bypass lock/gate controls.

Out of scope (for this posture doc): host compromise and arbitrary server code modification.

## Boundary Taxonomy

- Hard boundaries: server-enforced and fail-closed.
- Firm boundaries: server-enforced but intentionally overridable under policy.
- Soft boundaries: instructions/process only.

Target production posture: minimize firm boundaries and never depend on soft boundaries for safety.

## Control Set

## C1. Identity and Role Separation

- Autonomous loop identity MUST run with `FOUNDRY_MCP_ROLE=autonomy_runner`.
- Maintainer identity MUST be separate and manual-only.
- Observer identity SHOULD be used for dashboards/read-only tools.

Rationale:
- Prevent unattended loop from invoking maintainer-only escape hatches.

## C2. Escape Hatch Policy

Set and keep:
- `FOUNDRY_MCP_AUTONOMY_SECURITY_ALLOW_LOCK_BYPASS=false`
- `FOUNDRY_MCP_AUTONOMY_SECURITY_ALLOW_GATE_WAIVER=false`

Rationale:
- Converts write-lock/gate-waiver from firm to effectively closed during autonomous operation.

## C3. Required Gate Enforcement

Keep enabled:
- `FOUNDRY_MCP_AUTONOMY_SECURITY_ENFORCE_REQUIRED_PHASE_GATES=true`

Rationale:
- Prevents phase/spec completion when required gate obligations are unsatisfied.

## C4. Bounded Execution

Session config defaults for unattended runs:
- `stop_on_phase_completion=true`
- `gate_policy=strict`
- `auto_retry_fidelity_gate=true`
- bounded session limits (`max_consecutive_errors`, `max_tasks_per_session`, `max_fidelity_review_cycles_per_phase`)

Supervisor policy:
- Continue automatically only on `phase_complete`.
- Escalate on any non-phase pause, `failed`, or policy/integrity conflict.

## C5. Deterministic Escalation

Escalate and stop on:
1. `pause_reason=fidelity_cycle_limit`
2. `pause_reason=gate_failed`
3. `pause_reason=gate_review_required`
4. `pause_reason=blocked`
5. `pause_reason=error_threshold`
6. `ERROR_REQUIRED_GATE_UNSATISFIED`
7. `ERROR_GATE_AUDIT_FAILURE`
8. `ERROR_GATE_INTEGRITY_CHECKSUM`
9. repeated `ERROR_INVALID_GATE_EVIDENCE`
10. `FEATURE_DISABLED`/`AUTHORIZATION` for expected autonomy actions

Escalation packet SHOULD include:
- `spec_id`, `session_id`, `phase_id`
- `status`, `pause_reason`, `error_code`
- `last_step_id`, `last_step_type`
- counters (`consecutive_errors`, `fidelity_review_cycles`)
- recommended operator actions

## C6. Observability and Audit

Operator-visible state SHOULD be available from MCP surfaces:
- `task(action="session-status")`
- `task(action="session-list")`
- `task(action="progress")`
- `journal(action="list")`

Telemetry expectations:
- Authorization denials and rate-limit events
- Session lifecycle transitions
- Step outcomes and gate outcomes
- Any attempted bypass/waiver path (even if denied)

Operational polling recommendation:
- Poll `session-status` every 10-30 seconds during active runs.
- Snapshot summary after each primitive run.

## C7. Override Governance (Manual Path)

If an override is ever necessary:
1. Autonomous loop must be stopped.
2. Operator switches to maintainer identity.
3. Override must include structured reason code.
4. All override actions must be journaled/audited.
5. Autonomous loop resumes only after explicit operator approval.

Never allow unattended loop to self-authorize override paths.

## Posture Profiles

## strict-prod (recommended for unattended loops)

- Role: `autonomy_runner`
- Lock bypass: disabled
- Gate waiver: disabled
- Gate enforcement: enabled
- Loop continue rule: only `phase_complete`
- Any conflict: escalate + stop

## staging

- Same as strict-prod by default
- Allows policy tuning in a controlled environment
- Still prohibits unattended self-override

## debug

- Role: maintainer (manual only)
- Escape hatches may be enabled temporarily
- No unattended loops
- Mandatory reason codes and audit review

## Implementation Checklist

1. Configure runtime role to `autonomy_runner` for loop process.
2. Disable lock bypass and gate waiver.
3. Keep required gate enforcement enabled.
4. Enforce strict supervisor continue/stop matrix.
5. Implement escalation packet + operator handoff path.
6. Add alerting for integrity/audit/policy failures.
7. Validate with dry runs in staging before unattended production use.

## Validation Scenarios

The posture is acceptable when these tests pass:
1. Attempted bypass from loop identity is denied.
2. Attempted gate waiver from loop identity is denied.
3. Fidelity cycle limit pauses and triggers escalation.
4. Required gate unsatisfied blocks completion and escalates.
5. Feature-disabled and auth-denied states fail fast with actionable details.
6. Operator can observe real-time progress via status/journal/progress calls.

## Final Guidance

Treat skill instructions as guardrails, not controls.

The hard safety envelope should come from:
- role restrictions,
- disabled escape hatches,
- required-gate enforcement,
- deterministic supervisor escalation.

This gives you an autonomous loop that is productive when healthy and predictably stops when policy or integrity conditions are violated.
