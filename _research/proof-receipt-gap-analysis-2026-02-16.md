# P3 Proof/Receipt Gap Analysis

Date: 2026-02-16  
Scope: WS5 / P3 from `_research/PLAN.md` and `_research/PLAN-EXECUTION-CHECKLIST.md`

## Audit Inputs

- `src/foundry_mcp/core/autonomy/orchestrator.py`
- `src/foundry_mcp/core/autonomy/memory.py`
- `src/foundry_mcp/core/autonomy/models.py`
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py`
- `tests/unit/test_core/autonomy/test_orchestrator.py`
- `tests/unit/test_core/autonomy/test_handlers_session_step.py`
- `tests/unit/test_core/autonomy/test_memory.py`

## Pre-Remediation Findings

1. Step-proof scaffolding existed but was not enforced in runtime.
   - `step_proof` fields and proof storage helpers existed.
   - Session-step handlers/orchestrator did not enforce one-time proof consumption.
   - Replay/conflict semantics (`PROOF_CONFLICT` / `PROOF_EXPIRED`) were not exposed to callers.

2. Verification receipt validation was partial.
   - Required receipt checks existed for `execute_verification` success.
   - Field-level shape and binding checks were incomplete.
   - Not all receipt fields were validated for strict contract shape (`command_hash`, `output_digest`, timestamp constraints, task binding).

3. Integrity-failure semantics were not fully normalized.
   - Gate checksum failures were surfaced as generic `INVALID_GATE_EVIDENCE`.
   - This reduced operator clarity for integrity vs input-format failures.

## Remediation Scope Confirmed

The remediation scope for P3 was set to:

1. Enforce one-time `step_proof` consumption with deterministic replay/conflict/expiry semantics.
2. Tighten verification receipt contract validation and binding checks.
3. Emit explicit integrity-class error details for checksum/audit failure paths.
4. Document consumer receipt construction contract and signed-receipt deferral.

## Remediation Implemented

1. Step-proof enforcement + replay semantics:
   - Step issuance now includes one-time `step_proof` tokens on all emitted steps.
   - Session-step report path now consumes proof tokens exactly once and persists replayable response envelopes.
   - Deterministic outcomes added:
     - `STEP_PROOF_MISSING`
     - `STEP_PROOF_MISMATCH`
     - `STEP_PROOF_CONFLICT`
     - `STEP_PROOF_EXPIRED`

2. Verification receipt hardening:
   - Receipt model validation enforces SHA-256 digest shape and timezone-aware timestamps.
   - Orchestrator now validates receipt-task binding and receipt time window constraints.

3. Integrity failure normalization:
   - Gate checksum failures now return `GATE_INTEGRITY_CHECKSUM`.
   - Gate audit inconsistencies continue to return `GATE_AUDIT_FAILURE`.
   - Session-step error mapping includes actionable remediation for each integrity class.

## Signed Verification Receipts (Deferred)

Signed/cryptographic verification receipts remain deferred per PLAN Resolved Decision #4.  
Current P3 scope hardens existing proof and receipt enforcement paths without introducing signature infrastructure.

