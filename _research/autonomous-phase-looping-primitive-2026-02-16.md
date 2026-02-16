# Autonomous Phase Looping Primitive

Date: 2026-02-16
Context: `foundry-mcp` autonomy/session-step and fidelity-gate flows on branch `tyler/foundry-mcp-20260214-0833`

## Goal

Define how a one-phase primitive (`run_one_phase`) can be safely looped to spec completion, with clear behavior for edge cases and explicit human-escalation policy.

## Summary Opinion

Yes, this is a good primitive.

`run_one_phase(spec_id, workspace)` is a strong base because it is:
- Bounded: single phase max.
- Deterministic: explicit stop (`phase_complete` pause or terminal status).
- Quality-gated: phase boundary depends on fidelity-gate acceptance.
- Durable: session state persists for resume/recovery.

The right architecture is:
1. Keep `run_one_phase` strict and narrow.
2. Put spec-completion looping in a supervisor loop.
3. Continue automatically only on well-defined success states.
4. Escalate for ambiguous, policy, integrity, or repeated-failure states.

## Why This Fits Current Runtime

The branch has native support for this pattern:
- Session lifecycle + persistence: `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py`
- Step engine: `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py`
- Orchestrator stop condition (`stop_on_phase_completion`): `src/foundry_mcp/core/autonomy/orchestrator.py`
- Fidelity gate action/evidence: `src/foundry_mcp/tools/unified/review.py`
- Pause reasons include `phase_complete`, `fidelity_cycle_limit`, `gate_failed`, `gate_review_required`: `src/foundry_mcp/core/autonomy/models.py`

## Primitive Contract

## Function

`run_one_phase(spec_id, workspace) -> PhaseRunResult`

## Required session config

- `stop_on_phase_completion=true`
- `gate_policy=strict` (recommended default)
- `auto_retry_fidelity_gate=true` (recommended default)
- `enforce_autonomy_write_lock=true`

## Return categories

- `phase_complete`: success for one-phase primitive.
- `spec_complete`: overall spec done.
- `paused_needs_attention`: non-phase pause requiring decision.
- `failed`: hard failure.
- `blocked_runtime`: feature/auth/config incompatibility.

## Supervisor Loop to Spec Completion

Use a two-level model:
- Inner primitive: one phase, deterministic.
- Outer supervisor: loop until `spec_complete` or escalation.

```text
while true:
  result = run_one_phase(spec_id, workspace)

  if result.category == "spec_complete":
    return DONE

  if result.category == "phase_complete":
    continue  # next phase

  if result.category in {"paused_needs_attention", "failed", "blocked_runtime"}:
    escalate_to_human(result)
    return STOPPED
```

Important: only continue automatically on `phase_complete`.

## Edge Cases You Asked About

## 1) What if the phase is incomplete?

Treat as non-success for the primitive.

Typical signals:
- `status=paused` with reason not equal to `phase_complete`.
- session-step errors such as staleness, gate issues, required-gate unsatisfied.

Recommendation:
- Do not auto-continue outer loop.
- Return `paused_needs_attention` with full context and remediation options.

Rationale:
- Incomplete phase means the quality/sequence boundary was not satisfied.

## 2) What if a task within the phase is partially complete?

Model this as work-in-progress evidence, not task completion.

What runtime does:
- Only successful step outcomes for task/verification contribute to completion counters.
- Non-success outcomes do not mark completion and can lead to re-issuance/retry behavior.

Recommended primitive behavior:
- Report outcome `failure` (or `skipped`) with detailed `note` and `files_touched`.
- Preserve artifacts and progress notes.
- Allow bounded retries inside same phase cycle.
- Escalate when retry budget is exceeded or no forward progress is observed.

Suggested retry policy:
- `max_attempts_per_task_per_phase = 3`
- Escalate if same task emitted > 3 times without completion.

## 3) What if fidelity gate hits max iterations?

Current runtime behavior:
- Orchestrator pauses with `pause_reason=fidelity_cycle_limit` when phase cycle count reaches `max_fidelity_review_cycles_per_phase`.

Recommendation:
- Yes, stop the loop and escalate for human review.

Why:
- This is an explicit anti-spin safety stop.
- Continuing automatically defeats the purpose of cycle-limit guardrails.

## Escalation Policy Design

Use explicit escalation levels with deterministic triggers.

## L0 Auto-Recoverable (no human yet)

Allowed automatic recovery actions:
- Replay-safe retry (`session-step-replay`) for transient response-loss scenarios.
- Single retry for temporary operational issues.

Must remain bounded:
- `max_transient_retries = 1` or `2`.

## L1 Human Review Required

Stop and escalate for operator decision on:
- `pause_reason=fidelity_cycle_limit`
- `pause_reason=gate_failed`
- `pause_reason=gate_review_required`
- `pause_reason=blocked`
- `pause_reason=error_threshold`
- repeated same-task retries without progress
- `ERROR_REQUIRED_GATE_UNSATISFIED`

## L2 Maintainer/Security Escalation

Stop and escalate with high severity on:
- `ERROR_GATE_AUDIT_FAILURE`
- `ERROR_GATE_INTEGRITY_CHECKSUM`
- repeated `ERROR_INVALID_GATE_EVIDENCE`
- authorization denials for expected autonomy actions
- feature-flag mismatch in environments expected to support autonomy

## Continue vs Escalate Matrix

| Signal | Continue Loop? | Escalate? | Notes |
|---|---:|---:|---|
| `status=paused`, `pause_reason=phase_complete` | Yes | No | This is expected one-phase success |
| `status=completed` | No | No | Spec finished |
| `pause_reason=context_limit` | No | Usually Yes | Requires fresh invocation/context reset strategy |
| `pause_reason=heartbeat_stale` | No | Yes | Investigate caller health |
| `pause_reason=step_stale` / `ERROR_STEP_STALE` | No | Yes | Potential abandoned/disconnected execution |
| `pause_reason=task_limit` | No | Optional | Could be policy checkpoint; choose org policy |
| `pause_reason=blocked` | No | Yes | Dependencies/manual unblock needed |
| `pause_reason=gate_failed` | No | Yes | Quality boundary not met |
| `pause_reason=gate_review_required` | No | Yes | Manual policy requires acknowledgment |
| `pause_reason=fidelity_cycle_limit` | No | Yes (mandatory) | Anti-spin condition |
| `ERROR_SPEC_REBASE_REQUIRED` | No | Yes | Spec changed; requires rebase decision |
| `ERROR_REQUIRED_GATE_UNSATISFIED` | No | Yes | Invariant violation |
| `ERROR_GATE_AUDIT_FAILURE` | No | Yes (high) | Integrity/tamper suspicion |
| `ERROR_VERIFICATION_RECEIPT_*` | No | Yes | Proof/report contract mismatch |
| `FEATURE_DISABLED` / `AUTHORIZATION` | No | Yes | Environment/role misconfiguration |

## Additional Escalation Conditions Worth Adding

Even if not enforced by runtime today, the supervisor should escalate on:
1. No forward progress over N primitive runs.
- Example: phase unchanged for 2-3 consecutive runs.

2. Repeated same exception family.
- Example: same validation/integrity error 2+ times.

3. High-risk override requests.
- Any proposal to use gate waiver or lock bypass should require explicit human approval and reason code.

4. Drift between declared and observed capability shape.
- If discovery says action exists but runtime rejects deterministically, escalate config mismatch.

## Escalation Payload Design

When escalating, emit a structured packet for operator handoff:

```json
{
  "spec_id": "...",
  "session_id": "...",
  "phase_id": "...",
  "primitive_run_id": "...",
  "category": "paused_needs_attention|failed|blocked_runtime",
  "severity": "L1|L2",
  "reason_code": "fidelity_cycle_limit|gate_failed|...",
  "status": "paused|failed",
  "pause_reason": "...",
  "error_code": "...",
  "last_step_id": "...",
  "last_step_type": "...",
  "attempt_counters": {
    "task_attempts_in_phase": 0,
    "fidelity_cycles_in_phase": 0,
    "consecutive_errors": 0
  },
  "recommended_actions": [
    "..."
  ]
}
```

This enables deterministic incident response and fast human decisions.

## Recommended Human-Review Playbook

For each escalation class, provide a default action list:

- `fidelity_cycle_limit`:
  1. Inspect latest fidelity findings.
  2. Decide: targeted remediation vs phase/scope rewrite.
  3. Resume only after explicit strategy update.

- `partial task repeated`:
  1. Identify missing acceptance criteria or hidden dependency.
  2. Decide whether to split task or add dependency/requirement.
  3. Resume once plan is unblocked.

- `spec_rebase_required`:
  1. Review spec diff intent.
  2. Use rebase path if intentional structural change.
  3. Otherwise investigate unintended spec mutation.

- `gate_audit/integrity failures`:
  1. Freeze autonomous progression.
  2. Audit evidence/session files and recent overrides.
  3. Resume only after maintainer sign-off.

## Design Recommendations

1. Keep primitive strict:
- Success is only `phase_complete` or `spec_complete`.

2. Keep supervisor conservative:
- Only auto-continue on `phase_complete`.

3. Keep escalation explicit:
- Never silently downgrade policy/integrity failures.

4. Keep retries bounded:
- Small budgets for transient failures, then escalate.

5. Keep artifact hygiene high:
- Persist structured run summaries for every primitive invocation.

## Practical Default Policy

- `max_attempts_per_task_per_phase = 3`
- `max_transient_retries_per_step = 1`
- `max_consecutive_phase_runs_without_progress = 2`
- Escalate mandatory on:
  - `fidelity_cycle_limit`
  - `gate_review_required`
  - `gate_failed`
  - any integrity/audit failure
  - any required-gate unsatisfied conflict

## Final Take

Yes: one-phase, fidelity-gated execution is the right primitive.

To scale to full spec completion safely, use a supervisor loop that treats
`phase_complete` as the only auto-continue signal and escalates all other
non-terminal outcomes via a structured human-review path.

## Boundary Hardness Matrix

The instruction in the skill, `Do not manually bypass autonomy write locks`, is a **soft boundary**. It is useful guidance, but it is not a security boundary by itself.

### Hard vs Soft Boundary Model

| Boundary | Enforcement Layer | Hardness | Bypass Path | Recommended Posture |
|---|---|---|---|---|
| Feature gates (`autonomy_sessions`, `autonomy_fidelity_gates`) | Server handler entrypoints | Hard/fail-closed | Reconfigure server flags | Enable only in intended environments; fail startup if expected flags are missing |
| Role allowlists (`autonomy_runner`, `maintainer`, `observer`) | Server authorization dispatch | Hard/fail-closed | Start server with broader role | Run automation as `autonomy_runner`; separate maintainer identity |
| Step identity checks (`step_id`, `step_type`, bindings) | Orchestrator validation | Hard | None without changing server code/state | Keep mandatory |
| Gate evidence + receipt validation | Orchestrator validation | Hard | None without changing server code/state | Keep mandatory |
| Required-gate invariant + gate audit | Orchestrator completion guards | Hard | Waiver path if enabled | Disable waiver in autonomous posture |
| Write lock on task/lifecycle mutation | Server write-lock helper | Firm | `bypass_autonomy_lock=true` when policy allows | Set `allow_lock_bypass=false` in autonomous posture |
| Gate waiver | Privileged command path | Firm | `gate-waiver` when policy allows | Set `allow_gate_waiver=false` in autonomous posture |
| Skill prompt rule: “do not bypass” | Skill/instruction layer | Soft | Model/operator can ignore | Keep as defense-in-depth, never as primary control |
| Human process (“review before override”) | Operational process | Soft | Human non-compliance | Add explicit approval workflow + logging |

### Is the Boundary Too Soft?

It can be, if escape hatches are enabled and the caller has maintainer-level authority.

For safe unattended looping, make boundaries hard by policy:
1. Run loop identity as `autonomy_runner` only.
2. Keep `allow_lock_bypass=false`.
3. Keep `allow_gate_waiver=false`.
4. Treat any override/bypass attempt as immediate escalation + stop.

### Recommended Posture Profiles

| Profile | Intended Use | Role | Lock Bypass | Gate Waiver | Auto-Continue Rule |
|---|---|---|---|---|---|
| `strict-prod` | unattended/autonomous delivery | `autonomy_runner` | disabled | disabled | continue only on `phase_complete` |
| `staging` | validation and tuning | `autonomy_runner` | disabled (default) | disabled (default) | continue only on `phase_complete`; allow controlled experiments |
| `debug` | maintainer troubleshooting | `maintainer` | enabled only with reason code | enabled only with reason code | no unattended looping; operator-driven |

### Practical Design Rule

Use this ordering for controls:
1. **Server policy** (hard/fail-closed)
2. **Supervisor escalation logic** (deterministic stop)
3. **Skill instructions** (soft guidance)

If these disagree, server policy wins.
