---
name: foundry-implement-auto
description: Autonomous single-phase implementation using Foundry MCP session-step orchestration. Runs one spec phase under unattended posture with strict gate enforcement, deterministic exits, and verification receipts. Use when executing autonomous implementation with the autonomy_runner role and session-step protocol.
---

# foundry-implement-auto

## Purpose

Run exactly one autonomous implementation phase for a spec in unattended mode, then stop deterministically at the phase boundary or any escalation condition.

Reference:
- Runtime contract: `src/foundry_mcp/skills/foundry_implement_auto.py`
- Supervisor runbook: `docs/guides/autonomy-supervisor-runbook.md`
- Agent isolation: `docs/guides/autonomy-agent-isolation.md`

## Flow

> `[x?]`=decision `(GATE)`=server gate `→`=sequence `↻`=loop `!`=escalation `§`=section ref

```
Entry → [spec_id arg present?]
  → [no] → ! SPEC_ID_MISSING → EXIT
  → [yes] → §Preflight (6 checks)
  → [all pass?]
    → [no] → ! FoundryImplementAutoError + remediation → EXIT
    → [yes] → SessionStart
      → [SPEC_SESSION_EXISTS?]
        → [yes] → §SessionReuse → §StaleSessionRecovery (replay probe) → StepLoop
        → [no]  → StepLoop (fresh session, no stale steps possible)
  → StepLoop (max 200 iterations) ↻
    → task(session-step, command="next")
    → [loop_signal?]
      → [phase_complete] → EXIT success
      → [spec_complete] → EXIT success
      → [paused|failed|blocked_runtime] → ! EXIT escalation
      → [continue] → Dispatch(next_step.type)
        → [implement_task] → §ImplementTask
        → [execute_verification] → §ExecuteVerification
        → [run_fidelity_gate] → §RunFidelityGate
        → [address_fidelity_feedback] → §AddressFidelityFeedback
        → [pause|complete_spec] → EXIT
      → Report outcome → ↻ StepLoop
  → [max_iterations exceeded] → ! LOOP_LIMIT_EXCEEDED → EXIT
→ Emit exit packet
```

## MCP Tooling

All interactions use the Foundry MCP router+action pattern: `mcp__plugin_foundry_foundry-mcp__<router>`.

| Router | Key Actions |
|--------|-------------|
| `task` | `session` (start/list), `session-step` (next/report/replay), `prepare` |
| `spec` | `find` |
| `server` | `capabilities` |
| `review` | `fidelity-gate` |
| `verification` | `execute` |

## Non-Negotiable Rules

1. **Zero interaction.** Never use `AskUserQuestion` or any interactive prompt. This skill runs fully unattended. If any required input is missing (spec_id, environment), emit a structured error and EXIT immediately — do not ask, guess, or offer choices.
2. **Runtime truth only.** Use MCP tool responses as the source of truth for feature flags, posture, and role. Never trust discovery metadata or manifest alone.
3. **Server-driven sequencing.** The session-step orchestrator is the sole authority on task ordering, phase boundaries, and gate timing. Never infer or override task sequence.
4. **No privilege escalation.** Never set `allow_lock_bypass=true`, `allow_gate_waiver=true`, or request elevated role. Always pass `enforce_autonomy_write_lock=true` on session start.
5. **Fail fast on policy violations.** Any preflight failure terminates immediately with structured remediation.
6. **Deterministic exit.** Stop on `phase_complete` or `spec_complete`. Escalate on all other terminal `loop_signal` values. Never continue on ambiguous state.
7. **Bounded execution.** Hard limit of `max_iterations=200`. If exceeded without terminal signal, stop and escalate.

## Invocation

```sh
claude -p /foundry-implement-auto <spec-id> .
```

- Arg 1: `spec_id` — **required.** The spec to execute. If missing or empty, emit `SPEC_ID_MISSING` error and EXIT immediately. Never prompt, list specs, or offer choices.
- Arg 2: workspace path (default `.`).

**Argument extraction:** After skill expansion, the spec_id appears as text in the user's message — typically the remaining content after the skill name or command tag. To extract it:

1. **Look in the user message** for any text following `/foundry-implement-auto` or any `<command-name>` tag. The spec_id is the first whitespace-delimited token after the skill name.
2. **Scan the full conversation turn** for a string matching a spec ID pattern (hyphenated slug, e.g., `hello-world-python-2026-02-18-001` or `hello-world-python-2026-02-18-001.json`).
3. Strip any `.json` suffix if present (e.g., `hello-world-python-2026-02-18-001.json` → `hello-world-python-2026-02-18-001`).

If no spec_id is found after checking all locations, emit `SPEC_ID_MISSING` error and EXIT.

## Preflight

All checks must pass before session start. Any failure produces a structured error with error code and remediation hint. **Never prompt the user on failure — emit the error and EXIT.**

0. **Extract spec_id from the user's message.** Scan the user's message text in the current conversation turn for the spec_id (see Argument extraction above). If none found, emit `SPEC_ID_MISSING` error and EXIT. Do not list specs, offer choices, or ask for input.
1. Resolve spec via `spec(action="find")`
2. Detect action shape (canonical vs legacy)
3. Verify feature flags via `server(action="capabilities")`
4. Verify posture profile (reject `debug`, accept `unattended`/`supervised`)
5. Verify role authorization
6. Confirm shape detection and role verification are separate probes

> Full preflight sequence with error codes: [references/preflight.md](./references/preflight.md)

## Session Start

Start a session with hardened one-phase defaults:

```json
{
  "action": "session", "command": "start", "spec_id": "...",
  "gate_policy": "strict",
  "stop_on_phase_completion": true,
  "auto_retry_fidelity_gate": true,
  "enforce_autonomy_write_lock": true,
  "idempotency_key": "..."
}
```

These parameters are hardcoded — never configurable, never weakened.

> Session reuse and rebase handling: [references/session-management.md](./references/session-management.md)
> Stale session recovery: [references/session-management.md](./references/session-management.md#stale-session-recovery)

## Step Loop

Repeat up to `max_iterations=200`:

1. Call `task(action="session-step", command="next", session_id=...)`.
2. Read `data.loop_signal` — apply deterministic exit (see table below).
3. If no terminal signal, extract `data.next_step` and dispatch by `next_step.type`.
4. If `next_step.step_proof` is present, pass it through verbatim in `last_step_result.step_proof`.
5. Report outcome via simple report or extended `last_step_result` envelope.

> Step handler details: [references/step-handlers.md](./references/step-handlers.md)
> Verification receipt construction: [references/verification-receipts.md](./references/verification-receipts.md)
> Step proof protocol: [references/step-proofs.md](./references/step-proofs.md)

**Recovered-step entry:** When stale session recovery returns a pending step (see [references/session-management.md](./references/session-management.md#stale-session-recovery)), that step becomes the first iteration of the step loop. Dispatch it through the same step handler path (step 3 above) using the recovered `next_step` and `step_proof`. No special handling is needed — the step loop is agnostic to whether a step came from `next` or `replay`. The report response provides a fresh proof for subsequent iterations.

## Deterministic Exit Table

| `loop_signal` | Exit type | `final_status` | Action |
|---|---|---|---|
| `phase_complete` | Success | `paused` | Single phase done. Supervisor may queue next. |
| `spec_complete` | Success | `completed` | Spec fully implemented. |
| `paused_needs_attention` | Escalation | `paused` | Route to operator. |
| `failed` | Escalation | `failed` | Investigate before retry. |
| `blocked_runtime` | Escalation | varies | Resolve auth/feature/integrity/proof errors. |

## Error Handling

1. **Transport retries:** Use `task(action="session-step", command="replay")` for cached response without re-execution.
2. **Spec drift:** Surface `session-rebase` guidance to operator. Only `maintainer` role can rebase.
3. **No direct state mutation.** Never call `task(action="complete")` or `task(action="update")` directly — the orchestrator manages task state.

> Full error code taxonomy: [references/error-codes.md](./references/error-codes.md)

## Agent Isolation

The MCP server enforces authorization for MCP tool calls. Native Claude Code tools (Write, Edit, Bash) are constrained by caller hooks and environment configuration.

**Allowed:** Write/Edit to source files (`src/`, `tests/`), Bash for tests/linting, Read/Glob/Grep unrestricted, read-only git operations.

**Prohibited:** Writing to specs/config/sessions/audit, destructive git operations.

> Full isolation rules and guard scripts: [references/agent-isolation.md](./references/agent-isolation.md)

## Exit Payload

Always emit a complete exit packet:

- `spec_id`, `session_id`, `final_status`, `loop_signal`
- `pause_reason` (if paused)
- `active_phase_id`, `last_step_id`
- `recommended_actions` (if present — machine-readable escalation actions)
- `details.response_success`, `details.error_code`, `details.recommended_actions`
- Concise summary

The exit packet is the skill's only output. The supervisor uses it to decide next actions.
