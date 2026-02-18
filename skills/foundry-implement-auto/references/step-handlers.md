# Step Handlers

Dispatched by `next_step.type` from the step loop. Each handler reports outcome via the `last_step_result` envelope.

## Contents

- [Reporting Outcomes](#reporting-outcomes)
- [implement_task](#implement_task)
- [execute_verification](#execute_verification)
- [run_fidelity_gate](#run_fidelity_gate)
- [address_fidelity_feedback](#address_fidelity_feedback)
- [pause](#pause)
- [complete_spec](#complete_spec)

## Reporting Outcomes

**Always use the extended `last_step_result` envelope.** The simple `command="report"` transport cannot pass `task_id`, `phase_id`, `gate_attempt_id`, or `step_proof` — fields required by every step type. Using it will fail validation and waste the step proof.

Report by calling `session-step` with `command="next"` and the full `last_step_result` dict:

```
task(
  action="session-step",
  command="next",
  session_id="...",
  last_step_result={
    "step_id": "<from next_step.step_id>",
    "step_type": "<from next_step.type>",
    "outcome": "success|failure|skipped",
    "step_proof": "<from next_step.step_proof, verbatim>",
    "task_id": "<if implement_task or execute_verification>",
    "phase_id": "<if run_fidelity_gate or address_fidelity_feedback>",
    "gate_attempt_id": "<if run_fidelity_gate>",
    "note": "<optional summary>",
    "files_touched": ["<optional list of modified files>"],
    "verification_receipt": { "<if execute_verification>" }
  }
)
```

**Outcome values:** `"success"` | `"failure"` | `"skipped"`. No other values are accepted.

**Step proof:** If `next_step.step_proof` is present, include it **unchanged** in `last_step_result.step_proof`. Omitting it causes `STEP_PROOF_MISSING`. Altering it causes `STEP_PROOF_MISMATCH`. Both are fatal `blocked_runtime` errors.

## implement_task

1. Read task details from `next_step.instruction` — this is always available and is the primary context source.
2. Optionally call `task(action="prepare", spec_id=...)` for additional scope context (acceptance criteria, file hints).
3. Apply code changes to source files.
4. Run relevant checks (imports, syntax, basic tests).
5. Report outcome.

**Required `last_step_result` fields:** `step_id`, `step_type`, `outcome`, `task_id`, `step_proof`.

Example:

```json
{
  "step_id": "step-abc123",
  "step_type": "implement_task",
  "outcome": "success",
  "task_id": "task-1-1",
  "step_proof": "proof-xyz789",
  "note": "Created hello.py with main function and entry guard",
  "files_touched": ["hello.py"]
}
```

## execute_verification

1. Run the verification command from `next_step.instruction`, or call `verification(action="execute", ...)`.
2. Construct a verification receipt using the canonical helper. See [verification-receipts.md](./verification-receipts.md).
3. Include the receipt in `last_step_result.verification_receipt`.
4. If the receipt is missing or malformed, the server returns `ERROR_VERIFICATION_RECEIPT_MISSING` or `ERROR_VERIFICATION_RECEIPT_INVALID` (`blocked_runtime`).

**Required `last_step_result` fields:** `step_id`, `step_type`, `outcome`, `task_id`, `step_proof`, `verification_receipt`.

Example:

```json
{
  "step_id": "step-abc123",
  "step_type": "execute_verification",
  "outcome": "success",
  "task_id": "task-1-1",
  "step_proof": "proof-xyz789",
  "verification_receipt": {
    "command_hash": "a1b2c3...",
    "exit_code": 0,
    "output_digest": "d4e5f6...",
    "issued_at": "2026-01-15T10:30:00+00:00",
    "step_id": "step-abc123"
  }
}
```

## run_fidelity_gate

1. Call `review(action="fidelity-gate", spec_id=..., session_id=..., phase_id=..., step_id=...)`.
2. Capture `gate_attempt_id` from the response — this is **required** in the step report.
3. Report gate outcome.
4. Under strict gate policy, the server validates gate evidence integrity via checksum. A checksum failure returns `ERROR_GATE_INTEGRITY_CHECKSUM` (`blocked_runtime`).

**Required `last_step_result` fields:** `step_id`, `step_type`, `outcome`, `phase_id`, `gate_attempt_id`, `step_proof`.

Example:

```json
{
  "step_id": "step-abc123",
  "step_type": "run_fidelity_gate",
  "outcome": "success",
  "phase_id": "phase-1",
  "gate_attempt_id": "gate-abc123",
  "step_proof": "proof-xyz789"
}
```

## address_fidelity_feedback

1. Retrieve fidelity findings from `next_step.instruction`.
2. Remediate code and/or tests to address the findings.
3. Report changed files, outcome, and `phase_id`.
4. The orchestrator limits fidelity review cycles per phase (default 3). Exceeding the limit triggers `fidelity_cycle_limit` pause reason (`paused_needs_attention`).

**Required `last_step_result` fields:** `step_id`, `step_type`, `outcome`, `phase_id`, `step_proof`.

Example:

```json
{
  "step_id": "step-abc123",
  "step_type": "address_fidelity_feedback",
  "outcome": "success",
  "phase_id": "phase-1",
  "step_proof": "proof-xyz789",
  "note": "Fixed indentation and added missing docstring per gate feedback",
  "files_touched": ["src/utils.py", "tests/test_utils.py"]
}
```

## pause

Stop the loop and return an escalation packet with the pause reason. Do not attempt to continue or resolve the pause condition.

## complete_spec

Success stop — the spec is fully implemented. Return a success exit packet.
