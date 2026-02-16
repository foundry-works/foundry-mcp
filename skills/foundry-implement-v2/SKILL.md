---
name: foundry-implement-v2
description: Executes one autonomous implementation phase for a spec using Foundry MCP session-step orchestration, with deterministic loop-signal exits and verification receipt construction.
---

# foundry-implement-v2

## Purpose

Run exactly one autonomous implementation phase for a spec, then stop deterministically at the phase boundary or any escalation condition.

Reference runtime contract:
- `src/foundry_mcp/skills/foundry_implement_v2.py`

## Invocation Contract

- Expected invocation: `claude -p /foundry-implement-v2 <spec-id> .`
- Arg 1: `spec_id`
- Arg 2: workspace path (default `.`)

## Non-Negotiable Rules

1. Use MCP responses as runtime truth; treat discovery/manifest as hints.
2. Use session-step orchestration as the source of truth for sequencing.
3. Never bypass write locks or waive gates from autonomous flow.
4. Fail fast on role/feature preflight failures.
5. Treat runtime responses as truth; never trust manifest/discovery alone.
6. Stop on `phase_complete` or `spec_complete`; escalate all other `loop_signal` values.

## Startup Preflight

1. Resolve `spec_id` and workspace.
2. Validate spec exists:
   - `spec(action="find", spec_id=...)`
3. Detect action-shape compatibility:
   - Prefer canonical: `task(action="session", command=...)` and `task(action="session-step", command=...)`
   - Fallback: `session-start`, `session-step-next`, etc.
4. Verify runtime feature flags:
   - `server(action="capabilities")`
   - Fail fast if `runtime.autonomy.enabled_now.autonomy_sessions` is `false`.
   - Fail fast if fidelity-gate steps are required and `autonomy_fidelity_gates` is `false`.
   - Fail fast if `runtime.autonomy.posture_profile == "debug"` (debug posture is manual-only, not unattended).
5. Verify role/capability preflight:
   - Canonical: `task(action="session", command="list", limit=1)`
   - Legacy fallback: `task(action="session-list", limit=1)`
   - If `AUTHORIZATION` or `FEATURE_DISABLED`, stop with remediation.
6. Keep action-shape probe + role probe separate:
   - Probe determines canonical vs legacy compatibility.
   - Role probe is the authorization gate and must run before session start.

## Session Start

Start session with one-phase defaults:

- `task(action="session", command="start", spec_id=..., gate_policy="strict", stop_on_phase_completion=true, auto_retry_fidelity_gate=true, enforce_autonomy_write_lock=true, idempotency_key=...)`

If active non-terminal session exists for spec:

- Reuse only when it is compatible with one-phase run semantics.
- Otherwise stop and require explicit operator intent (never force-end silently).

## Step Loop

Repeat:

1. `task(action="session-step", command="next", session_id=...)`
2. Read `data.loop_signal` and apply deterministic exits:
   - `phase_complete`: success stop (single phase done)
   - `spec_complete`: success stop
   - `paused_needs_attention`: escalation stop
   - `failed`: escalation stop
   - `blocked_runtime`: escalation stop
3. If continuing, execute `data.next_step` by `next_step.type`.
4. Report outcome via:
   - Preferred: `task(action="session-step", command="report", session_id=..., step_id=..., step_type=..., outcome=...)`
   - For extended payloads (`verification_receipt`, `gate_attempt_id`, `step_proof`, `task_id`, `phase_id`), use:
     - `task(action="session-step", command="next", session_id=..., last_step_result=...)`
   - Rationale: `session-step-report` is an alias optimized for simple reports.

## Step Handlers

### `implement_task`

1. Use `task(action="prepare", spec_id=...)` as needed.
2. Apply code changes.
3. Run relevant checks.
4. Report `last_step_result` with `task_id`, `note`, and `files_touched`.

### `execute_verification`

1. Run verification command or `verification(action="execute", ...)`.
2. Construct `verification_receipt` with required fields:
   - `command_hash`
   - `exit_code`
   - `output_digest`
   - `issued_at`
   - `step_id`
3. Include receipt in `last_step_result`.
4. If any step includes `step_proof`, pass it through unchanged in `last_step_result`.

### `run_fidelity_gate`

1. Call `review(action="fidelity-gate", spec_id=..., session_id=..., phase_id=..., step_id=...)`.
2. Capture `gate_attempt_id`.
3. Report gate outcome in `last_step_result`.

### `address_fidelity_feedback`

1. Retrieve fidelity findings.
2. Remediate code/tests.
3. Report changed files and outcome.

### `pause` and `complete_spec`

- `pause`: stop and return escalation packet.
- `complete_spec`: success stop.

## Deterministic Exit Table

- `phase_complete` -> success stop (`final_status=paused`)
- `spec_complete` -> success stop
- `paused_needs_attention` -> escalation stop
- `failed` -> escalation stop
- `blocked_runtime` -> escalation stop

## Error Handling

1. For retry-safe transport repeats, use `task(action="session-step", command="replay", session_id=...)`.
2. On state drift requiring reconciliation, surface `session-rebase` guidance.
3. Never mutate task state outside session-step protocol to unstick flow.

## Exit Payload

Always emit:

- `spec_id`
- `session_id`
- `final_status`
- `loop_signal`
- `pause_reason` (if paused)
- `active_phase_id`
- `last_step_id`
- concise summary
