# Error Code Taxonomy

All structured errors include `code`, `message`, and `remediation` fields.

## Contents

- [Startup Errors](#startup-errors)
- [Session Errors](#session-errors)
- [Step Dispatch Errors](#step-dispatch-errors)
- [Loop Errors](#loop-errors)
- [Runtime Block Errors](#runtime-block-errors)

## Startup Errors

| Code | Phase | Cause |
|------|-------|-------|
| `SPEC_ID_MISSING` | Preflight 0 | No spec_id argument provided in invocation |
| `SPEC_RESOLUTION_FAILED` | Preflight 1 | Spec not found |
| `ACTION_SHAPE_UNSUPPORTED` | Preflight 2 | Neither canonical nor legacy action shape accepted |
| `CAPABILITIES_UNAVAILABLE` | Preflight 3 | Server capabilities endpoint unreachable |
| `FEATURE_DISABLED` | Preflight 3 | Required feature flag not enabled |
| `POSTURE_UNSUPPORTED` | Preflight 4 | Posture profile is `debug` or unrecognized |
| `AUTHORIZATION` | Preflight 5 | Role lacks session access |

## Session Errors

| Code | Cause |
|------|-------|
| `SESSION_START_FAILED` | Session start rejected by server |
| `SESSION_REUSE_AMBIGUOUS` | Multiple non-terminal sessions found for spec |
| `SESSION_REUSE_INCOMPATIBLE` | Existing session missing hardened settings |

## Step Dispatch Errors

| Code | Cause |
|------|-------|
| `STEP_TYPE_UNSUPPORTED` | Unknown `next_step.type` value |
| `GATE_ATTEMPT_REQUIRED` | Fidelity gate report missing `gate_attempt_id` |
| `VERIFICATION_RECEIPT_REQUIRED` | Verification report missing receipt |

## Loop Errors

| Code | Cause |
|------|-------|
| `STEP_MISSING` | Server returned success but no `next_step` or `loop_signal` |
| `LOOP_LIMIT_EXCEEDED` | `max_iterations` (200) exhausted without terminal signal |

## Runtime Block Errors

These map to `blocked_runtime` loop signal and trigger immediate escalation:

| Code | Cause |
|------|-------|
| `AUTHORIZATION` | Role authorization failed mid-session |
| `FEATURE_DISABLED` | Feature flag changed mid-session |
| `ERROR_REQUIRED_GATE_UNSATISFIED` | Phase gate not satisfied |
| `ERROR_GATE_AUDIT_FAILURE` | Gate audit trail validation failed |
| `ERROR_GATE_INTEGRITY_CHECKSUM` | Gate evidence checksum mismatch |
| `ERROR_INVALID_GATE_EVIDENCE` | Gate evidence malformed |
| `ERROR_VERIFICATION_RECEIPT_MISSING` | Receipt not provided |
| `ERROR_VERIFICATION_RECEIPT_INVALID` | Receipt field validation failed |
| `STEP_PROOF_MISSING` | Proof token not included |
| `STEP_PROOF_MISMATCH` | Proof token doesn't match |
| `STEP_PROOF_CONFLICT` | Proof already consumed |
| `STEP_PROOF_EXPIRED` | Proof grace window elapsed |
