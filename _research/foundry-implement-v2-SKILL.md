---
name: foundry-implement-v2
description: Executes one autonomous implementation phase for a spec using Foundry MCP session-step orchestration, including fidelity-gate enforcement and deterministic stop at phase completion. Use for headless runs like `claude -p /foundry-implement-v2 <spec-id> .`.
---

# foundry-implement-v2

## Purpose

Run exactly one phase of autonomous implementation for a spec in headless mode, stop only after the phase boundary is reached and the fidelity gate outcome is accepted by policy.

## Invocation Contract

Expected invocation:
- `claude -p /foundry-implement-v2 <spec-id> .`

Interpretation:
- Arg 1: `spec_id`
- Arg 2: workspace path (default `.`)

## Non-Negotiable Rules

1. Use MCP only; do not parse spec JSON directly from disk.
2. Use session-step orchestration as the source of truth for sequencing.
3. Do not manually bypass autonomy write locks.
4. Fail fast on disabled autonomy features or authorization denials.
5. Stop at first successful phase boundary pause (`pause_reason=phase_complete`).

## Startup Preflight

1. Resolve `spec_id` and workspace.
2. Validate spec exists: `spec(action="find", spec_id=...)`.
3. Discover task action compatibility via `server(action="tools")`:
- Prefer unified intents if available (`session`, `session-step`).
- Else use concrete actions (`session-start`, `session-step-next`, etc.).
4. Verify autonomy features by issuing a lightweight session call and checking for `FEATURE_DISABLED`.
5. If denied with `AUTHORIZATION`, stop and surface required role/action.

## Session Start

Start a session for `spec_id` with:
- `gate_policy="strict"`
- `stop_on_phase_completion=true`
- `auto_retry_fidelity_gate=true`
- `enforce_autonomy_write_lock=true`
- optional: `idempotency_key` for retry-safe starts

If an active non-terminal session already exists for the same spec:
- Reuse/resume if compatible with one-phase run semantics.
- Otherwise stop and ask for explicit operator intent (do not force-end silently).

## Main Execution Loop

Repeat until terminal condition:

1. Call `session-step-next`.
2. If response status is `paused`:
- If `pause_reason=phase_complete`: exit success.
- Else exit paused with remediation details.
3. If response status is `completed`: exit success.
4. Read `next_step` and execute by `next_step.type`.
5. Report outcome with `session-step-report` (or `session-step-next` + `last_step_result`).

## Step Handlers

### implement_task

1. Load task context with `task(action="prepare", spec_id=...)` when needed.
2. Implement code changes in workspace.
3. Run relevant local checks.
4. Report:
- `step_id`, `step_type="implement_task"`, `task_id`, `phase_id`
- `outcome` (`success|failure|skipped`)
- `note`, `files_touched`

### execute_verification

1. Execute verification using `verification(action="execute", spec_id=..., verify_id=...)`.
2. Build and include `verification_receipt` in step result when outcome is success.
3. Report:
- `step_type="execute_verification"`
- `verification_receipt` with required fields (`command_hash`, `exit_code`, `output_digest`, `issued_at`, `step_id`)

### run_fidelity_gate

1. Call:
- `review(action="fidelity-gate", spec_id, session_id, phase_id, step_id, ...)`
2. Capture returned `gate_attempt_id`.
3. Report:
- `step_type="run_fidelity_gate"`
- `gate_attempt_id`
- `phase_id`, `outcome`

### address_fidelity_feedback

1. Retrieve fidelity findings from gate artifacts.
2. Implement remediation edits.
3. Run relevant tests/checks.
4. Report outcome with notes and touched files.

### pause / complete_spec

- `pause`: stop immediately and return pause reason/remediation.
- `complete_spec`: stop successfully.

## Error Handling

On deterministic orchestration errors (e.g., step mismatch, stale step, invalid gate evidence):
1. Prefer `session-step-replay` for safe retry when appropriate.
2. If state drift is detected, surface `session-rebase` guidance.
3. Do not mutate task state outside session-step flow to “unstick” execution.

## Exit Conditions

Successful stop for one-phase mode:
- Session status `paused` with `pause_reason=phase_complete`.

Other stop conditions:
- `paused` for non-phase reasons (return remediation)
- `failed` (return failure reason and next action)
- unrecoverable authorization/feature-gate errors

## Minimal Output Contract

At end of run, report:
- `spec_id`
- `session_id`
- `final_status`
- `pause_reason` (if paused)
- `active_phase_id`
- `last_step_id`
- concise summary of work completed in this run
