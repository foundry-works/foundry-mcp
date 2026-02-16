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

4. Document convention: treat discovery/manifest metadata as hints; actual tool responses are the source of truth for runtime-enabled state. This applies to all consumers, including the v2 skill preflight (see WS8).

### Likely Files

- `src/foundry_mcp/config.py`
- `src/foundry_mcp/core/discovery.py`
- `src/foundry_mcp/tools/unified/server.py`
- `samples/foundry-mcp.toml`

### Tests

1. Config parsing tests for `[feature_flags]` and env override precedence.
2. Capability endpoint tests for runtime-accurate feature states.
3. Test that capability response diverges from manifest when flags are toggled at runtime.

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

### Loop Signal Mapping Reference

The mapping table below defines the canonical mapping. Implementations MUST produce these exact signals for the given conditions. The five signals are intentionally coarse — escalation detail belongs in `recommended_actions`, not in signal proliferation.

| Condition | `loop_signal` |
|---|---|
| `pause_reason=phase_complete` | `phase_complete` |
| Session status `completed` / `pause_reason=spec_complete` | `spec_complete` |
| `pause_reason` in {`fidelity_cycle_limit`, `gate_failed`, `gate_review_required`, `blocked`, `error_threshold`, `context_limit`, `heartbeat_stale`, `step_stale`, `task_limit`, `spec_rebase_required`} | `paused_needs_attention` |
| Step/session status `failed`, unrecoverable step error | `failed` |
| `ERROR_REQUIRED_GATE_UNSATISFIED`, `ERROR_GATE_AUDIT_FAILURE`, `ERROR_GATE_INTEGRITY_CHECKSUM`, feature-disabled, authorization denial, repeated `ERROR_INVALID_GATE_EVIDENCE` | `blocked_runtime` |

Supervisor rule: auto-continue **only** on `phase_complete`. All other signals stop the loop and emit an escalation payload.

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

2. Add session event feed as a journal-backed filtered view:
- `task(action="session-events", session_id=..., cursor=..., limit=...)`
- Implementation: filtered query over existing journal entries scoped to session, not new persistence.
- Rationale: journal already captures lifecycle/step/gate events. Adding a new event store is unjustified infrastructure until concrete operational evidence shows journal is insufficient.

3. Standardize event payload shape for dashboards.

### Performance Constraint

Design target: up to 10 concurrent autonomous sessions, each polled at 10-30s intervals, with journal volumes up to 10,000 entries per session. The journal-backed event feed query must remain under 200ms at this scale. If benchmarks during P2 development show this is not met, add a short-TTL read cache (30-60s) for `session-status` and indexed journal queries for `session-events` before merging — not as a post-merge follow-up.

### Likely Files

- `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py`
- `src/foundry_mcp/core/pagination.py` usage where needed
- Journal query paths (no new persistence module)

### Tests

1. Pagination + cursor tests for journal-backed event feeds.
2. Snapshot tests for session-status expanded fields.

### Acceptance Criteria

1. Operator can monitor loop health from MCP APIs without file inspection.
2. Event feed uses existing journal persistence with stable pagination and backward-compatible envelopes.

## WS5. Integrity/Proof Hardening Completion

### Problem

Proof-related structures exist, but enforcement posture depends on partial runtime paths. The existing step-identity + replay-cache provides the primary integrity guarantee; step-proof consumption and receipt validation need to be tightened on those existing paths.

### Scope Decision

Signed verification receipts (cryptographic signatures) are **deferred** to a future hardening phase. The current threat model (see `autonomous-posture-hardening-2026-02-16.md`) excludes host compromise and arbitrary server code modification, which are the scenarios where cryptographic signing provides incremental value over the existing receipt validation. This cycle focuses on making the existing paths airtight.

### Changes

0. **Audit existing paths first.** Before remediation, produce a written gap analysis of current proof/receipt enforcement. Identify: which fields are validated, which are accepted unchecked, which consumption semantics are enforced vs. aspirational. P3 remediation scope is defined by this audit — acceptance criteria are conditional on its findings.

1. Tighten step-proof consumption enforcement on existing paths:
- one-time token consumption
- replay grace behavior
- deterministic conflict response

2. Harden verification receipt lifecycle:
- ensure issuance/validation semantics are consistent and auditable
- close any gaps where receipt fields are accepted but not validated
- document the receipt construction contract (`command_hash`, `exit_code`, `output_digest`, `issued_at`, `step_id`) so that consumers (including the v2 skill, WS8) can produce valid receipts

3. Audit all integrity-failure paths for deterministic escalation-class responses.

### Likely Files

- `src/foundry_mcp/core/autonomy/orchestrator.py`
- `src/foundry_mcp/core/autonomy/memory.py`
- `src/foundry_mcp/core/autonomy/models.py`

### Tests

1. Replay/duplicate/report conflict tests.
2. Integrity checksum and audit-failure contract tests.
3. Receipt validation boundary tests (field completeness, binding correctness).

### Acceptance Criteria

1. Existing proof/receipt enforcement is complete and has no unchecked acceptance paths.
2. Integrity failures are explicit, deterministic, and non-silent.

## WS6. Posture Profiles + Policy Validation

### Problem

Hardening settings are present but not bundled into operator-friendly posture profiles.

### Changes

1. Introduce posture profiles:
- `unattended` — autonomous runner, all escape hatches closed, headless-safe
- `supervised` — human in the loop, standard guardrails, escape hatches available with reason codes
- `debug` — maximum flexibility, no unattended loops, mandatory audit trail
- Profiles are a fixed enumeration, not extensible. Operators who need custom behavior configure individual flags directly. This prevents profile sprawl and keeps the posture surface auditable.

2. Profile-driven defaults for:
- role
- `allow_lock_bypass`
- `allow_gate_waiver`
- gate enforcement and recommended limits

3. Startup posture validator:
- reject or warn on unsafe combos (for example unattended + maintainer + bypass enabled)

4. Add role verification preflight convention:
- document a lightweight pattern (e.g., `task(action="session", command="list")`) for consumers to verify role/capability before starting a session
- if the preflight returns an authorization denial, fail fast with remediation guidance
- this addresses the risk that role restrictions silently block flow mid-session

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

5. Deprecation warning mechanism:
- Emit a `deprecated` field in the response envelope metadata for legacy action invocations (not just server-side logs). This ensures machine consumers can programmatically detect and migrate.
- Include `deprecated.action`, `deprecated.replacement`, `deprecated.removal_target` (date or version) in the metadata.
- Log a `WARN`-level message server-side for operator visibility.

### Acceptance Criteria

1. Docs and manifest are contract-accurate.
2. CI coverage includes loop semantics and posture controls.
3. Migration path is explicit and time-bounded.
4. Legacy action invocations include machine-readable deprecation metadata in the response.

## WS8. V2 Skill Integration + End-to-End Validation

### Problem

WS1-WS7 build infrastructure for autonomous phase looping, but the primary consumer — the `foundry-implement-v2` skill — is not tracked as a deliverable. Without it, the infrastructure is untestable against its intended use case.

### Inputs

- `_research/foundry-implement-v2-SKILL.md` (drafted skill definition)
- `_research/foundry-implement-v2-research-2026-02-16.md` (compatibility analysis)

### Changes

1. Land the `foundry-implement-v2` skill definition based on the drafted SKILL.md.

2. Implement skill startup preflight:
- spec resolution and validation
- action-shape compatibility detection (canonical vs. legacy)
- feature flag verification (fail fast on disabled `autonomy_sessions` or `autonomy_fidelity_gates`)
- role/capability verification via lightweight session call (see WS6 item 4)

3. Implement the step-driven execution loop:
- `session-step-next` consumption
- step-type dispatch (`implement_task`, `execute_verification`, `run_fidelity_gate`, `address_fidelity_feedback`, `pause`, `complete_spec`)
- `session-step-report` with proper verification receipt construction (see WS5 receipt contract)

4. Implement deterministic exit:
- stop on `phase_complete` (success)
- stop on `spec_complete` (success)
- stop and report on all other pause reasons or failures

5. End-to-end validation:
- run the skill against a test spec in `unattended` posture
- verify the full loop: preflight -> session start -> step loop -> fidelity gate -> phase boundary stop
- verify escalation: simulate each `loop_signal` category and confirm correct skill exit behavior

### Likely Files

- `skills/foundry-implement-v2/SKILL.md`
- Integration test fixtures

### Tests

1. Preflight tests (feature flags disabled, role denied, action-shape detection).
2. Step loop unit tests (each step type handler).
3. End-to-end integration test against a minimal test spec.
4. Exit behavior tests for each `loop_signal` category.

### Acceptance Criteria

1. The v2 skill can run a single phase of a test spec to completion in `unattended` posture.
2. Preflight fails fast with actionable guidance when prerequisites are unmet.
3. All six research caveats (A-F) are addressed in the implementation.

## 6. Phased PR Plan

## P0a (Contract Foundation): Action Shape Reconciliation

Scope:
1. WS1 canonical session/session-step action shape with aliases.
2. Manifest/docs alignment for action shapes only.

Deliverables:
1. Canonical action paths merged with legacy aliases.
2. Backward compatibility confirmed.

Gate to merge:
1. Both canonical and legacy invocations pass.
2. Manifest/docs match runtime for action shapes.

Rationale for split: WS1 and WS2 share no code paths. Smaller PR, faster review, earlier confidence signal.

## P0b (Config Foundation): Feature Flags + Capability Truth

Scope:
1. WS2 runtime feature flag config and capability truth.

Deliverables:
1. Feature flag config wiring merged.
2. Capability endpoint reports runtime-enabled state.

Gate to merge:
1. Feature flag config parsing tests pass.
2. Capability truthfulness tests pass.

## P1 (Supervisor Semantics): Loop Signal + Escalation Schema

Depends on: P0a, P0b (canonical action shapes and feature flag config must be merged first — `loop_signal` mapping references action shapes and feature-flag error conditions).

Scope:
1. WS3 `loop_signal` + `recommended_actions` fields.
2. `loop_signal` on step responses only (see Resolved Decisions #1).
3. Derived `session_signal` summary on `session-status` if operators need it.
4. Mapping table from Section 5 WS3 is the authoritative reference for implementation.

Deliverables:
1. Deterministic mapping implementation.
2. Contract tests for all stop/continue branches.

Gate to merge:
1. No regression in existing response fields.
2. New fields are additive-only (see Rollback Strategy).

## P2 (Observability): Session Status + Events

Depends on: P1 (`session_signal` on session-status derives from P1's `loop_signal` definitions).

Scope:
1. WS4 expanded status fields.
2. Journal-backed session event feed with pagination (no new persistence).
3. Performance validation against WS4 performance constraint (200ms target at design-scale).

Deliverables:
1. Operator-facing monitorable session APIs.

Gate to merge:
1. Pagination + envelope tests.

## P3 (Hardening Completion): Proof/Receipt Integrity

Scope:
1. WS5 audit of existing proof/receipt paths (produces gap analysis before remediation).
2. Remediation of gaps identified by audit (scope determined by audit findings).
3. Integrity escalation standardization.
4. Receipt construction contract documented for consumer use (signed receipts deferred — see Resolved Decisions #4).

Deliverables:
1. Written gap analysis of current proof/receipt enforcement.
2. Deterministic integrity behavior and tests on identified gaps.
3. Documented receipt construction contract.

Gate to merge:
1. Gap analysis reviewed and remediation scope confirmed.
2. Replay/conflict/receipt suites green.
3. Receipt construction contract is documented and testable.

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
2. Emit deprecation warnings for legacy action names (deprecation window: 3 months or 2 minor releases, whichever comes later — see Resolved Decisions #3).
3. Deprecation warnings use machine-readable response envelope metadata (`deprecated.action`, `deprecated.replacement`, `deprecated.removal_target`), not just server-side logs — see WS7 item 5.

Deliverables:
1. Release-ready docs + changelog entries.
2. Deprecation warnings active on legacy action paths with machine-readable metadata.

Gate to merge:
1. Review checklist completion and docs verification.
2. Legacy action invocations include `deprecated` metadata in response envelope.

## P6 (Consumer Validation): V2 Skill + End-to-End

Depends on: P0a-P5 (all infrastructure phases landed).

Scope:
1. WS8 v2 skill implementation.
2. End-to-end validation against a test spec in `unattended` posture.
3. Verification that all six research caveats (A-F) are addressed.

Deliverables:
1. `foundry-implement-v2` skill landed and functional.
2. End-to-end test demonstrating: preflight -> session start -> step loop -> fidelity gate -> phase boundary stop.
3. Escalation behavior validated for each `loop_signal` category.

Gate to merge:
1. V2 skill completes a single phase of a test spec in `unattended` posture.
2. Preflight rejects gracefully when prerequisites are unmet.
3. Each `loop_signal` category triggers correct skill exit behavior.

## 7. Detailed Task Backlog by PR

> **Note:** These are tracking-level items. The following will need further decomposition into implementation subtasks before work begins:
> - **P0a items 1-2:** Canonical dispatch adapters — need to determine normalization location and routing changes.
> - **P1 items 1-2:** `loop_signal` model integration and mapping helper — need to identify all response paths that must emit the field.
> - **P3 item 1:** Proof path audit — scope of remediation depends on audit output.
> - **P6 items 1-3:** V2 skill implementation — step handler details depend on finalized contracts from P0a-P5.
>
> Use spec tooling to break down before starting each phase.

## P0a Task List (Action Contract)

1. Implement canonical `task(action="session", command=...)` dispatch adapter.
2. Implement canonical `task(action="session-step", command=...)` dispatch adapter.
3. Register alias mapping from legacy `session-*` / `session-step-*` actions.
4. Add deprecation metadata to legacy alias paths.
5. Update manifest and tool reference docs for action shapes.
6. Add unit + integration compatibility tests (canonical and legacy parity).

## P0b Task List (Config + Capability Truth)

1. Wire `[feature_flags]` config loading support.
2. Add env-based override support with documented precedence.
3. Ensure `server(action="capabilities")` reports runtime-enabled state.
4. Add startup validation warnings for inconsistent flag states.
5. Document "discovery as hints, responses as truth" convention.
6. Add config parsing and capability truthfulness tests (including runtime divergence from manifest).

## P1 Task List

1. Add `loop_signal` field to step response models.
2. Implement deterministic mapping helper per the mapping table in WS3 (status/pause_reason/error -> `loop_signal`).
3. Add `recommended_actions` schema for escalation cases.
4. Add derived `session_signal` summary to `session-status` (if needed).
5. Add contract tests for each mapping branch (must cover all rows in the WS3 mapping table).
6. Update docs and examples with supervisor-loop usage.

## P2 Task List

1. Extend session-status output fields (`last_step_id`, `last_step_type`, `current_task_id`, phase progress, retry counters).
2. Add `session-events` action as journal-backed filtered view with pagination.
3. Standardize event payload shape.
4. Add tests and operator examples.

## P3 Task List

1. Audit existing proof/receipt paths: produce written gap analysis (which fields validated, which unchecked, which consumption semantics enforced vs. aspirational).
2. Remediate gaps identified by audit: tighten consumption semantics and close unchecked acceptance paths.
3. Document receipt construction contract (`command_hash`, `exit_code`, `output_digest`, `issued_at`, `step_id`) for consumer use.
4. Standardize integrity error details for operator actionability.
5. Add replay/conflict/integrity tests.
6. Add receipt construction contract tests (valid and invalid receipt shapes).

## P4 Task List

1. Add posture config model and parsing (fixed enumeration: `unattended`, `supervised`, `debug` — not extensible).
2. Implement posture expansion to existing security flags + role defaults.
3. Add startup posture validation and logging.
4. Document and implement role verification preflight pattern for consumer use.
5. Add tests for unattended/supervised/debug behavior.
6. Update sample config and runbooks.

## P5 Task List

1. Enable deprecation warnings on legacy action name paths via response envelope metadata (`deprecated.action`, `deprecated.replacement`, `deprecated.removal_target`).
2. Add server-side `WARN`-level logging for legacy action invocations.
3. Add migration notes with deprecation timeline (3 months / 2 minor releases).
4. Update changelog and release notes.
5. Run checklist and ensure all referenced docs are updated.
6. Add tests verifying deprecation metadata presence in legacy action responses.

## P6 Task List (V2 Skill + End-to-End)

1. Land `foundry-implement-v2` skill definition (based on `_research/foundry-implement-v2-SKILL.md`).
2. Implement startup preflight: spec resolution, action-shape compatibility, feature flag verification, role/capability verification.
3. Implement step-driven execution loop: `session-step-next` -> dispatch -> `session-step-report`.
4. Implement step handlers: `implement_task`, `execute_verification` (with receipt construction), `run_fidelity_gate`, `address_fidelity_feedback`, `pause`, `complete_spec`.
5. Implement deterministic exit behavior for each `loop_signal` category.
6. Add preflight tests (disabled flags, denied role, action-shape detection).
7. Add step loop unit tests (each handler).
8. Add end-to-end integration test: full phase completion against a test spec in `unattended` posture.
9. Add escalation exit tests: simulate each non-`phase_complete` signal and verify correct skill exit.

## 8. Risk Register

1. Contract drift risk:
- Mitigation: contract tests + manifest/docs in same PR.

2. Backward compatibility risk:
- Mitigation: alias path + deprecation warnings before removal.

3. Security regression risk:
- Mitigation: fail-closed defaults; posture validation; authz tests.

4. Operator confusion risk:
- Mitigation: explicit `loop_signal`; recommended actions; runbooks; WS4 observability surfaces (session-status, session-events) for real-time monitoring.

5. Observability bloat risk:
- Mitigation: compact payload defaults + paginated event feed.

## 9. Rollback Strategy

All new response fields (`loop_signal`, `recommended_actions`, expanded session-status fields, event feed) are **additive-only**. Existing consumers that do not read these fields are unaffected by their presence.

Rollback approach by phase:
1. **P0a/P0b:** Legacy action aliases remain functional throughout. If canonical actions introduce regressions, disable them and revert to legacy-only routing. Feature flags are already gated — disable via config to restore prior behavior.
2. **P1:** `loop_signal` is a new field on step responses. Removing it does not break existing consumers. If the mapping logic introduces regressions in existing fields, revert the mapping helper only.
3. **P2:** Expanded session-status fields and the event feed action are new surfaces. Revert by removing the fields/action without affecting existing session-status behavior.
4. **P3:** Proof path tightening could reject previously-accepted inputs. If tightened validation causes false rejections, relax specific checks and investigate before re-tightening. Integrity checks should never be fully disabled — only adjusted.
5. **P4:** Posture profiles set defaults but do not remove existing per-flag configuration. Revert by removing profile expansion and reverting to direct flag configuration.
6. **P6:** The v2 skill is a consumer, not infrastructure. Reverting it has zero impact on the MCP server or other consumers. If end-to-end tests reveal infrastructure issues, fix them in the relevant infrastructure phase — do not ship workarounds in the skill.

## 10. Validation Strategy

1. Ship P0/P1 behind compatibility-first behavior.
2. Validate posture profiles by running test specs under each profile (`unattended`, `supervised`, `debug`) and confirming expected enforcement behavior.
3. Enable in selected internal workflows with `unattended` posture.
4. Gather telemetry on escalations and false positives.
5. Complete P2-P4 before recommending broad unattended usage.
6. Land P6 (v2 skill) and validate end-to-end before declaring the autonomy surface ready for unattended operation.

## 11. Acceptance Criteria (Program-Level)

1. A headless supervisor can safely run:
- loop while `loop_signal=phase_complete`
- stop/escalate otherwise

2. Runtime capability/feature outputs are truthful.

3. Operators can observe progress and diagnose stop conditions from MCP APIs alone.

4. Escape hatches are controlled by explicit posture policy, not prompt discipline.

5. All new/changed contracts are documented and test-covered.

6. The `foundry-implement-v2` skill can complete a single phase of a test spec end-to-end in `unattended` posture, validating that all infrastructure changes serve their intended consumer.

7. All six research caveats (A: action-shape, B: feature flags, C: role preflight, D: receipt construction, E: discovery vs. runtime, F: step-proof enforcement) are addressed and tested.

## 12. Resolved Decisions

1. **`loop_signal` placement:** Step responses only. The orchestrator emits outcomes at the step level; that's where the signal belongs. `session-status` may carry a derived `session_signal` summary if operators need a quick poll target, but the authoritative field lives on step responses.

2. **`session-events` implementation:** Journal-backed filtered view, not new persistence. Journal already captures lifecycle, step, and gate events. A new event store is unjustified infrastructure until concrete operational evidence shows journal is insufficient. If journal query performance becomes a bottleneck, add indexing or caching — don't add a parallel store.

3. **Deprecation window for legacy action names:** 3 months or 2 minor releases, whichever comes later. Deprecation warnings emitted from P0a merge onward. Legacy names removed after the window closes in a dedicated cleanup PR.

4. **Signed verification receipts:** Deferred to a future hardening phase. The current threat model excludes host compromise and arbitrary server code modification — the scenarios where cryptographic signing provides incremental value. This cycle tightens existing receipt validation paths instead.

5. **Deprecation warning mechanism:** Machine-readable metadata in the response envelope (`deprecated.action`, `deprecated.replacement`, `deprecated.removal_target`), not just server-side logs. MCP consumers are machines; they need programmatic detection.

6. **Posture profile extensibility:** Fixed enumeration only (`unattended`, `supervised`, `debug`). Custom behavior uses direct flag configuration. This keeps the posture surface auditable and prevents profile sprawl.

7. **Phase dependencies:** P1 depends on P0a+P0b. P2 depends on P1. P6 depends on P0a-P5. All other phases are independent.

## 13. Immediate Next Steps

1. Open a tracking spec for this plan (if not already represented).
2. Decompose P0a task items into implementation subtasks via spec tooling.
3. Start P0a: canonical action adapter + alias compatibility + manifest alignment.
4. Follow with P0b: feature flag config loading + capability truth response.
5. Land docs/manifest updates in the same PR as each respective phase.
6. After P0a-P5 land, start P6: v2 skill implementation + end-to-end validation.

