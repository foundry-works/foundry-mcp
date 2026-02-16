# MCP Autonomy Looping + Hardening Plan

Date: 2026-02-16
Owner: foundry-mcp core
Status: Draft
Inputs:
- `_research/foundry-implement-v2-research-2026-02-16.md`
- `_research/autonomous-phase-looping-primitive-2026-02-16.md`
- `_research/autonomous-posture-hardening-2026-02-16.md`
Execution Checklist: `./PLAN-EXECUTION-CHECKLIST.md`


## 1. Objective

Introduce a production-ready autonomy surface that supports:
1. Deterministic one-phase execution primitive (`run_one_phase`) that can be looped to spec completion.
2. Clear machine-readable continue/stop/escalate semantics.
3. Strong posture controls where server policy, not skill text, defines hard boundaries.

## 2. Desired End State

At end of this plan:
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

## WS1. Session API Contract Reconciliation

### Problem

Discovery/manifest indicate intent for `task(action="session", command=...)` and `task(action="session-step", command=...)`, while runtime currently relies on concrete action names (`session-start`, `session-step-next`, etc.).

### Changes

1. Add canonical handlers:
- `task(action="session", command="start|status|pause|resume|rebase|end|list|reset")`
- `task(action="session-step", command="next|report|replay|heartbeat")`

2. Keep current concrete actions as aliases during deprecation:
- `session-start` -> `session + command=start`
- `session-step-next` -> `session-step + command=next`
- etc.

3. Centralize normalization in one place:
- `src/foundry_mcp/tools/unified/task_handlers/__init__.py`
- Possibly helper in `src/foundry_mcp/tools/unified/task_handlers/_helpers.py`

### Contract/Docs

1. Update `mcp/capabilities_manifest.json` action/parameter docs to match runtime.
2. Update `docs/05-mcp-tool-reference.md` examples to canonical shapes.

### Tests

1. Add compatibility tests that both canonical and legacy action names route identically.
2. Add deprecation metadata checks for legacy names.

### Acceptance Criteria

1. Both canonical and legacy invocations pass.
2. Tool discovery and docs exactly reflect accepted runtime shape.

## WS2. Runtime Feature Flags + Capability Truthfulness

### Problem

Autonomy handlers are gated by `config.feature_flags`, but config-loading surfaces do not clearly expose robust flag management for runtime/operator workflows.

### Changes

1. Add explicit config support for feature flags:
- TOML section support (e.g., `[feature_flags]`)
- optional env-based overrides with clear precedence

2. Ensure capability responses reflect runtime state:
- `server(action="capabilities")` should include actual enablement of autonomy features.
- Maintain distinction between "supported by binary" and "enabled now".

3. Add startup validation warnings/errors for inconsistent flag states.

### Likely Files

- `src/foundry_mcp/config.py`
- `src/foundry_mcp/core/discovery.py`
- `src/foundry_mcp/tools/unified/server.py`
- `samples/foundry-mcp.toml`

### Tests

1. Config parsing tests for `[feature_flags]` and env override precedence.
2. Capability endpoint tests for runtime-accurate feature states.

### Acceptance Criteria

1. Operators can enable/disable autonomy flags through documented config paths.
2. Capability outputs are accurate for current runtime.

## WS3. Loop Outcome + Escalation Semantics (First-Class)

### Problem

Supervisors currently infer behavior from `status`, `pause_reason`, and error codes. Outcome semantics are present but not normalized into a single contract field.

### Changes

1. Add normalized `loop_signal` in relevant responses:
- `phase_complete`
- `spec_complete`
- `paused_needs_attention`
- `failed`
- `blocked_runtime`

2. Add `recommended_actions` payload for escalation cases.

3. Add deterministic mapping table in code from status/reason/error -> `loop_signal`.

### Likely Files

- `src/foundry_mcp/core/autonomy/models.py`
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py`
- `src/foundry_mcp/tools/unified/task_handlers/handlers_session_step.py`

### Tests

1. Contract tests for each `loop_signal` branch.
2. Regression tests for existing clients relying on old fields.

### Acceptance Criteria

1. Supervisors can continue loop by checking one field (`loop_signal`).
2. Escalation payloads are machine-readable and consistent.

## WS4. Operator Observability Surfaces

### Problem

Progress is observable but fragmented across session status, step responses, progress, and journal.

### Changes

1. Extend `session-status` response with operator-centric fields:
- `last_step_id`, `last_step_type`
- `current_task_id`
- `active_phase_progress`
- retry counters per phase/task (if available)

2. Add session events feed endpoint/action:
- candidate: `task(action="session-events", session_id=..., cursor=..., limit=...)`
- fallback: enrich and paginate existing journal/session metadata

3. Standardize event payload shape for dashboards.

### Likely Files

- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py`
- `src/foundry_mcp/core/autonomy/memory.py` (if event persistence added)
- `src/foundry_mcp/core/pagination.py` usage where needed

### Tests

1. Pagination + cursor tests for event feeds.
2. Snapshot tests for session-status expanded fields.

### Acceptance Criteria

1. Operator can monitor loop health from MCP APIs without file inspection.
2. Event/feed API supports stable pagination and backward-compatible envelopes.

## WS5. Integrity/Proof Hardening Completion

### Problem

Proof-related structures exist, but enforcement posture may still depend on partial runtime paths.

### Changes

1. Finalize step-proof consumption enforcement where intended:
- one-time token consumption
- replay grace behavior
- deterministic conflict response

2. Tighten verification receipt lifecycle:
- ensure issuance/validation semantics are consistent and auditable
- evaluate signed receipt extension (if feasible in this phase)

3. Audit all integrity-failure paths for deterministic escalation-class responses.

### Likely Files

- `src/foundry_mcp/core/autonomy/orchestrator.py`
- `src/foundry_mcp/core/autonomy/memory.py`
- `src/foundry_mcp/core/autonomy/models.py`

### Tests

1. Replay/duplicate/report conflict tests.
2. Integrity checksum and audit-failure contract tests.
3. Receipt validation boundary tests.

### Acceptance Criteria

1. Proof/receipt enforcement is complete for defined threat model.
2. Integrity failures are explicit, deterministic, and non-silent.

## WS6. Posture Profiles + Policy Validation

### Problem

Hardening settings are present but not bundled into operator-friendly posture profiles.

### Changes

1. Introduce posture profiles:
- `strict-prod`
- `staging`
- `debug`

2. Profile-driven defaults for:
- role
- `allow_lock_bypass`
- `allow_gate_waiver`
- gate enforcement and recommended limits

3. Startup posture validator:
- reject or warn on unsafe combos (for example unattended + maintainer + bypass enabled)

### Likely Files

- `src/foundry_mcp/config.py`
- `src/foundry_mcp/server.py`
- possibly new module under `src/foundry_mcp/core/autonomy/` for posture validation
- `samples/foundry-mcp.toml`

### Tests

1. Profile parsing + default expansion tests.
2. Startup validation tests (safe vs unsafe combinations).

### Acceptance Criteria

1. Operators can set one posture value and get predictable policy behavior.
2. Unsafe autonomous setups are blocked or loudly warned.

## WS7. Documentation + Testing + Migration

### Problem

Changes above impact contracts, behavior, and operational runbooks.

### Changes

1. Update docs:
- `docs/05-mcp-tool-reference.md`
- autonomy architecture docs/ADR references
- operator runbook for loop + escalation

2. Update manifest/examples:
- `mcp/capabilities_manifest.json`

3. Expand tests across unit/integration/contract suites.

4. Add migration notes + deprecation timeline for legacy action names.

### Acceptance Criteria

1. Docs and manifest are contract-accurate.
2. CI coverage includes loop semantics and posture controls.
3. Migration path is explicit and time-bounded.

## 6. Phased PR Plan

## P0 (Foundation): Contract + Config Truth

Scope:
1. WS1 canonical session/session-step action shape with aliases.
2. WS2 runtime feature flag config and capability truth.

Deliverables:
1. Canonical action paths merged.
2. Feature flag config wiring merged.
3. Manifest/docs aligned.

Gate to merge:
1. Backward compatibility tests pass.
2. Capability truth tests pass.

## P1 (Supervisor Semantics): Loop Signal + Escalation Schema

Scope:
1. WS3 `loop_signal` + `recommended_actions` fields.

Deliverables:
1. Deterministic mapping implementation.
2. Contract tests for all stop/continue branches.

Gate to merge:
1. No regression in existing response fields.

## P2 (Observability): Session Status + Events

Scope:
1. WS4 expanded status fields.
2. Session event feed action with pagination.

Deliverables:
1. Operator-facing monitorable session APIs.

Gate to merge:
1. Pagination + envelope tests.

## P3 (Hardening Completion): Proof/Receipt Integrity

Scope:
1. WS5 proof lifecycle completion.
2. Integrity escalation standardization.

Deliverables:
1. Deterministic integrity behavior and tests.

Gate to merge:
1. Replay/conflict/receipt suites green.

## P4 (Posture): Profile System + Startup Validation

Scope:
1. WS6 posture profiles.
2. Startup validator and runbook updates.

Deliverables:
1. One-knob posture selection for operators.

Gate to merge:
1. Unsafe unattended profiles rejected/warned as designed.

## P5 (Docs + Cleanup): Migration Closure

Scope:
1. WS7 final docs and migration notes.
2. Optional begin deprecation warnings for legacy action names.

Deliverables:
1. Release-ready docs + changelog entries.

Gate to merge:
1. Review checklist completion and docs verification.

## 7. Detailed Task Backlog by PR

## P0 Task List

1. Implement `session` and `session-step` command dispatch adapter.
2. Register/validate alias mapping from legacy actions.
3. Wire `[feature_flags]` config loading and env override behavior.
4. Ensure `server(action="capabilities")` reports runtime-enabled state.
5. Update manifest and tool reference docs.
6. Add unit + integration compatibility tests.

## P1 Task List

1. Add `loop_signal` model field(s).
2. Implement mapping helper in session/session-step response builders.
3. Add `recommended_actions` schema for escalation cases.
4. Add contract tests for each mapping branch.
5. Update docs and examples with supervisor-loop usage.

## P2 Task List

1. Extend session-status output fields.
2. Add session events persistence/read path (or journal-backed alternative).
3. Add `session-events` action and pagination support.
4. Add tests and operator examples.

## P3 Task List

1. Audit proof record path and enforce consumption semantics end-to-end.
2. Tighten/verify verification receipt issuance + validation coupling.
3. Standardize integrity error details for operator actionability.
4. Add replay/conflict/integrity tests.

## P4 Task List

1. Add posture config model and parsing.
2. Implement posture expansion to existing security flags + role defaults.
3. Add startup posture validation and logging.
4. Add tests for strict-prod/staging/debug behavior.
5. Update sample config and runbooks.

## P5 Task List

1. Add migration/deprecation notes for legacy action names.
2. Update changelog and release notes.
3. Run checklist and ensure all referenced docs are updated.

## 8. Risk Register

1. Contract drift risk:
- Mitigation: contract tests + manifest/docs in same PR.

2. Backward compatibility risk:
- Mitigation: alias path + deprecation warnings before removal.

3. Security regression risk:
- Mitigation: fail-closed defaults; posture validation; authz tests.

4. Operator confusion risk:
- Mitigation: explicit `loop_signal`; recommended actions; runbooks.

5. Observability bloat risk:
- Mitigation: compact payload defaults + paginated event feed.

## 9. Rollout Strategy

1. Ship P0/P1 behind compatibility-first behavior.
2. Validate in staging with `strict-prod` posture simulation.
3. Enable in selected internal workflows.
4. Gather telemetry on escalations and false positives.
5. Complete P2-P4 before recommending broad unattended usage.

## 10. Acceptance Criteria (Program-Level)

1. A headless supervisor can safely run:
- loop while `loop_signal=phase_complete`
- stop/escalate otherwise

2. Runtime capability/feature outputs are truthful.

3. Operators can observe progress and diagnose stop conditions from MCP APIs alone.

4. Escape hatches are controlled by explicit posture policy, not prompt discipline.

5. All new/changed contracts are documented and test-covered.

## 11. Open Decisions Needed

1. Should `loop_signal` be added to both `session` and `session-step` responses or only step responses?
2. Should `session-events` be a new action or a filtered view over existing journal/session artifacts?
3. What deprecation window should be used for legacy `session-*` action names?
4. Do we want signed verification receipts in this cycle or defer to a follow-up hardening phase?

## 12. Immediate Next Steps

1. Open a tracking spec for this plan (if not already represented).
2. Start P0 with a small vertical slice:
- canonical action adapter + compatibility tests
- feature flag config loading
- capability truth response
3. Land docs/manifest updates in the same PR as P0 contract behavior.

