# MCP Autonomy Looping + Hardening Plan

Date: 2026-02-16
Owner: foundry-mcp core
Status: Executed (P0a–P6 landed)

## 1. Objective

Introduce a production-ready autonomy surface that supports:
1. Deterministic one-phase execution primitive (`run_one_phase`) that can be looped to spec completion.
2. Clear machine-readable continue/stop/escalate semantics.
3. Strong posture controls where server policy, not skill text, defines hard boundaries.

## 2. Desired End State

1. API contracts and runtime action surfaces match.
2. Feature flags and capabilities are runtime-accurate and configurable.
3. Supervisors can reliably loop only on `phase_complete` and escalate on all other non-success states.
4. Operators can observe real-time progress/session health without inspecting raw state files.
5. Escape hatches are governable and auditable via posture profiles.
6. Tests and docs enforce and describe all of the above.

## 3. Non-Goals

1. Full multi-tenant zero-trust architecture.
2. Replacing the current autonomy state store backend.
3. Major redesign of non-autonomy routers unrelated to session/step orchestration.

## 4. Guiding Principles

1. Fail closed for policy and security-sensitive paths.
2. Preserve backward compatibility via explicit alias/deprecation windows.
3. Keep machine-readable outcomes first-class.
4. Keep operator workflows explicit; no silent policy downgrades.
5. Specs, docs, tests move together in each PR slice.

## 5. Workstreams

### WS1. Session API Contract Reconciliation

Add canonical `task(action="session", command=...)` and `task(action="session-step", command=...)` handlers. Keep legacy concrete actions (`session-start`, `session-step-next`, etc.) as deprecating aliases.

### WS2. Runtime Feature Flags + Capability Truthfulness

Wire explicit config support for feature flags (TOML `[feature_flags]` + env overrides). Ensure `server(action="capabilities")` reports runtime-enabled state, not just what the binary supports. Convention: discovery/manifest are hints; tool responses are truth.

### WS3. Loop Outcome + Escalation Semantics

Add normalized `loop_signal` field on step responses: `phase_complete`, `spec_complete`, `paused_needs_attention`, `failed`, `blocked_runtime`. Add `recommended_actions` payload for escalation cases. Deterministic mapping from status/pause_reason/error → signal.

### WS4. Operator Observability Surfaces

Extend `session-status` with operator-centric fields. Add journal-backed `session-events` feed with pagination (no new persistence). Design target: 10 concurrent sessions, 10k journal entries per session, queries under 200ms.

### WS5. Integrity/Proof Hardening Completion

Audit and tighten step-proof consumption and verification receipt validation on existing paths. Document receipt construction contract. Signed receipts deferred to a future cycle.

### WS6. Posture Profiles + Policy Validation

Fixed posture enumeration: `unattended`, `supervised`, `debug`. Profile-driven defaults for role, lock bypass, gate waiver, gate enforcement. Startup validator rejects unsafe combinations (e.g., unattended + maintainer + bypass enabled).

### WS7. Documentation + Testing + Migration

Docs, manifest, and tests updated in lockstep with each workstream. Deprecation warnings emitted as machine-readable response envelope metadata. Deprecation window: 3 months or 2 minor releases.

### WS8. V2 Skill Integration + End-to-End Validation

Land `foundry-implement-v2` skill with startup preflight, step-driven execution loop, and deterministic exit. Validate end-to-end against a test spec in `unattended` posture.

## 6. Resolved Decisions

1. **`loop_signal` placement:** Step responses only. `session-status` may carry a derived summary, but the authoritative field lives on step responses.
2. **`session-events` implementation:** Journal-backed filtered view, not new persistence.
3. **Deprecation window:** 3 months or 2 minor releases, whichever comes later.
4. **Signed verification receipts:** Deferred. Current threat model excludes host compromise.
5. **Deprecation warning mechanism:** Machine-readable metadata in response envelope, not just server-side logs.
6. **Posture profile extensibility:** Fixed enumeration only. Custom behavior uses direct flag configuration.
7. **Phase dependencies:** P1 depends on P0a+P0b. P2 depends on P1. P6 depends on P0a-P5.

## 7. Acceptance Criteria (Program-Level)

1. A headless supervisor can safely loop on `phase_complete` and stop/escalate otherwise.
2. Runtime capability/feature outputs are truthful.
3. Operators can observe progress and diagnose stop conditions from MCP APIs alone.
4. Escape hatches are controlled by explicit posture policy, not prompt discipline.
5. All new/changed contracts are documented and test-covered.
6. The `foundry-implement-v2` skill can complete a single phase end-to-end in `unattended` posture.
7. All six research caveats (A–F) are addressed and tested.
