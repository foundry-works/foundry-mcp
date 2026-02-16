# foundry-implement-v2 Research Memo

Date: 2026-02-16
Workspace analyzed: `/home/tyler/.sandboxes/worktrees/foundry-mcp-tyler-foundry-mcp-20260214-0833`

## Original Request

You asked for a reimagined `foundry-implement` skill as `foundry-implement-v2`, intended for headless invocation:

`claude -p /foundry-implement-v2 <spec-id> .`

Your target routine:
1. Fetch spec.
2. Find next phase via MCP tools.
3. Work until the phase is complete.
4. Ensure work passes fidelity gate.
5. Stop.

You asked two questions:
1. Is this consistent with the MCP changes on this branch?
2. If yes, what should the v2 skill look like?

## Assistant Elaboration of Your Request

I interpreted this as designing a single-phase autonomous control loop that uses the new autonomous session APIs as the source of truth, not manual task orchestration. The goal is for the skill to act as an orchestrator around MCP step emissions (`session-step-*`) and fidelity evidence (`review action=\"fidelity-gate\"`), then terminate deterministically at the phase boundary.

## Research Method

I validated behavior by reading implementation and contract surfaces in the current branch, including:
- `dev_docs/architecture/adr-002-autonomous-spec-execution.md`
- `specs/pending/autonomous-spec-execution-2026-02-14-001.json`
- `specs/pending/autonomy-boundary-hardening-2026-02-15-001.json`
- `src/foundry_mcp/tools/unified/task_handlers/__init__.py`
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py`
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py`
- `src/foundry_mcp/core/autonomy/orchestrator.py`
- `src/foundry_mcp/tools/unified/review.py`
- `src/foundry_mcp/core/autonomy/write_lock.py`
- `src/foundry_mcp/core/authorization.py`
- `src/foundry_mcp/tools/unified/server.py`
- `mcp/capabilities_manifest.json`
- Existing skill references:
  - `~/GitHub/claude-foundry/skills/foundry-implement/SKILL.md`
  - `~/GitHub/claude-foundry/skills/foundry-implement/references/autonomous-mode.md`
  - `~/GitHub/claude-foundry/skills/foundry-implement/references/session-management.md`

## Branch-Grounded Findings

### 1. Your proposed loop is fundamentally aligned

The branch implements durable autonomous sessions and step orchestration:
- Session lifecycle handlers exist (`session-start`, `session-status`, `session-pause`, `session-resume`, `session-end`, `session-list`, `session-rebase`, `session-reset`).
- Step engine exists (`session-step-next`, `session-step-report`, `session-step-replay`, `session-step-heartbeat`).
- Fidelity gate evidence path exists (`review action=\"fidelity-gate\"`).

### 2. Stop-at-phase-boundary behavior is explicitly supported

`session-start` accepts `stop_on_phase_completion`, and orchestrator step 15 pauses with `phase_complete` once gate conditions pass for the phase. This is exactly the control needed for “do one phase, pass gate, stop.”

### 3. Fidelity-gated progression is first-class

The orchestrator emits `run_fidelity_gate` steps and validates submitted evidence (`gate_attempt_id`, session/phase/step binding, integrity checksum) before allowing progression.

### 4. Session-step is the correct hot path

The orchestrator is the sequencing authority. A v2 skill should consume next steps and report outcomes rather than trying to infer phase/task sequencing itself.

### 5. Write-lock changes task mutation expectations

With an active non-terminal autonomous session, many direct task mutations are write-locked unless bypassed. This means v2 should avoid relying on out-of-band task mutation paths during autonomy runs.

### 6. Authorization and hardening are now central

Role-based allowlists and denial rate-limiting exist. `autonomy_runner` is scoped to autonomy session/step and fidelity-gate operations.

## Caveats and Gaps

### A. Action-shape mismatch between intent/docs and runtime naming

- Discovery and manifest indicate `task action=\"session\"` and `task action=\"session-step\"` are intended.
- Runtime task router currently exposes concrete action names like `session-start` and `session-step-next`.

Implication for v2:
- The skill should use a compatibility adapter at startup to detect which action surface is active and route calls accordingly.

### B. Feature flags are fail-closed and required

Session and session-step handlers reject when `autonomy_sessions` is not enabled; fidelity-gate rejects when `autonomy_fidelity_gates` is disabled.

Implication for v2:
- Add strict preflight and fail fast with actionable remediation if flags are off.

### C. Role requirements can block the flow

Authorization denies actions outside role allowlists.

Implication for v2:
- Preflight should verify role/capability by issuing a harmless session status/list call and reading deterministic authorization errors.

### D. Verification receipt requirement is strict at step reporting time

`execute_verification` success reports are validated for `verification_receipt` in `last_step_result`.

Implication for v2:
- The skill must construct and include the required receipt fields when reporting successful verification steps.

### E. Capability metadata may not perfectly represent runtime switches

Some server capability surfaces are static descriptors rather than guaranteed runtime-enabled state.

Implication for v2:
- Treat `server(action=\"tools\")`/`server(action=\"capabilities\")` as discovery hints; treat actual tool responses as runtime truth.

### F. Step-proof scaffolding appears partially implemented

Proof record plumbing exists in persistence models/storage, but orchestration flow currently relies primarily on step identity + replay cache behavior.

Implication for v2:
- Do not depend on strict step-proof enforcement semantics unless verified enabled in your deployment.

## Direct Answer to Your Two Questions

### Q1: Is your imagined flow consistent with branch changes?

Yes, with caveats A-F above.

The core idea (headless autonomous, phase completion, fidelity gate, deterministic stop) is consistent with this branch’s orchestration model.

### Q2: What should `foundry-implement-v2` look like?

It should be a strict orchestration skill with:
1. Runtime compatibility detection for session action naming.
2. Preflight for feature flags and role authorization.
3. Session start configured for `stop_on_phase_completion=true`.
4. A step loop driven by `session-step-next` + `session-step-report`.
5. Fidelity gate execution using `review fidelity-gate` when instructed.
6. Deterministic stop when paused at `phase_complete`.

## Proposed SKILL.md (Draft)

```md
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
```

## Standalone SKILL.md File

A standalone version of the same proposed skill is written to:
- `~/GitHub/foundry-mcp/_research/foundry-implement-v2-SKILL.md`

