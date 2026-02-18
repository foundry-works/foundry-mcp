# Step Handlers

Dispatched by `next_step.type` from the step loop. Each handler reports outcome via one of two transports.

## Contents

- [Report Transports](#report-transports)
- [implement_task](#implement_task)
- [execute_verification](#execute_verification)
- [run_fidelity_gate](#run_fidelity_gate)
- [address_fidelity_feedback](#address_fidelity_feedback)
- [pause](#pause)
- [complete_spec](#complete_spec)

## Report Transports

| Transport | When to use | Call |
|-----------|------------|------|
| Simple report | No extended fields needed | `task(action="session-step", command="report", session_id=..., step_id=..., step_type=..., outcome=..., note=..., files_touched=...)` |
| Extended report | Includes `verification_receipt`, `gate_attempt_id`, `step_proof`, `task_id`, or `phase_id` | `task(action="session-step", command="next", session_id=..., last_step_result={...})` |

Rationale: `session-step-report` is optimized for small payloads; extended fields require the full `last_step_result` envelope.

**Outcome values:** `"success"` | `"failure"` | `"skipped"`. No other values are accepted.

## implement_task

1. Read task details from `next_step.instruction` — this is always available and is the primary context source.
2. Optionally call `task(action="prepare", spec_id=...)` for additional scope context (acceptance criteria, file hints).
3. Apply code changes to source files.
4. Run relevant checks (imports, syntax, basic tests).
5. Report `last_step_result` with `task_id`, `note`, and `files_touched`.
   - `outcome`: `"success"` | `"failure"` | `"skipped"`

## execute_verification

1. Run the verification command from `next_step.instruction`, or call `verification(action="execute", ...)`.
2. Construct a verification receipt using the canonical helper. See [verification-receipts.md](./verification-receipts.md).
3. Include the receipt in `last_step_result.verification_receipt`.
4. If the receipt is missing or malformed, the server returns `ERROR_VERIFICATION_RECEIPT_MISSING` or `ERROR_VERIFICATION_RECEIPT_INVALID` (`blocked_runtime`).

## run_fidelity_gate

1. Call `review(action="fidelity-gate", spec_id=..., session_id=..., phase_id=..., step_id=...)`.
2. Capture `gate_attempt_id` from the response — this is **required** in the step report.
3. Report gate outcome in `last_step_result` with `gate_attempt_id` and `phase_id`.
4. Under strict gate policy, the server validates gate evidence integrity via checksum. A checksum failure returns `ERROR_GATE_INTEGRITY_CHECKSUM` (`blocked_runtime`).

## address_fidelity_feedback

1. Retrieve fidelity findings from `next_step.instruction`.
2. Remediate code and/or tests to address the findings.
3. Report changed files, outcome, and `phase_id`.
4. The orchestrator limits fidelity review cycles per phase (default 3). Exceeding the limit triggers `fidelity_cycle_limit` pause reason (`paused_needs_attention`).

## pause

Stop the loop and return an escalation packet with the pause reason. Do not attempt to continue or resolve the pause condition.

## complete_spec

Success stop — the spec is fully implemented. Return a success exit packet.
