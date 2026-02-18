# Step Proof Handling

The orchestrator includes a `step_proof` token on emitted steps. This is a one-time-use token for replay protection, enforced under unattended posture.

## Protocol

1. If `next_step.step_proof` is present, include it **unchanged** in `last_step_result.step_proof`.
2. The skill passes proof tokens through verbatim. It does not generate, modify, or validate proofs — that is the server's responsibility.

## Server Enforcement

The server enforces one-time consumption with a grace window for transport retries.

| Error Code | Cause |
|------------|-------|
| `STEP_PROOF_MISSING` | Proof required but not included in report |
| `STEP_PROOF_MISMATCH` | Proof does not match expected token |
| `STEP_PROOF_CONFLICT` | Proof already consumed (double-submit) |
| `STEP_PROOF_EXPIRED` | Proof grace window has elapsed |

All proof errors map to `blocked_runtime` loop signal and trigger immediate escalation.
