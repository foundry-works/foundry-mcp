# PLAN Execution Checklist

Date: 2026-02-16
Source Plan: `./PLAN.md`
Last Updated: 2026-02-16 (P0a-P6 complete; cross-phase safety + rollback readiness evidence recorded; completion criteria validated)

## How to Use

1. Work top-to-bottom by phase (`P0a` -> `P6`).
2. Do not start a new phase until the merge gate for current phase is satisfied.
3. Respect phase dependencies: P1 requires P0a+P0b. P2 requires P1. P6 requires P0a-P5.
4. Mark each item with `[x]` only after code, docs, and tests are complete.

## Program Readiness

- [ ] Tracking spec/issue created and linked
- [ ] Owners assigned for `P0a`-`P6`
- [ ] Validation environments identified (test specs and posture profiles for each profile: `unattended`, `supervised`, `debug`)
- [ ] Success metrics agreed (loop success rate, escalation rate, integrity incidents)
- [ ] Resolved Decisions 1-7 reviewed and confirmed (see PLAN.md Section 12)

Program-readiness verification tasks (pending ops/program management):
1. Create and link tracking issue/spec in this checklist (evidence target: issue URL + ID).
2. Add accountable owner per phase (`P0a`-`P6`) in checklist or linked tracker (evidence target: assignee matrix).
3. Record validation environment matrix for all postures (`unattended`, `supervised`, `debug`) (evidence target: environment table + test fixture IDs).
4. Publish success-metric baselines and thresholds (evidence target: dashboard/query links for loop success, escalations, integrity incidents).
5. Record explicit sign-off for Resolved Decisions 1-7 review (evidence target: dated reviewer sign-off note).

## P0a Contract Foundation: Action Shape Reconciliation

### Build

- [x] Implement canonical `task(action="session", command=...)` dispatch adapter
- [x] Implement canonical `task(action="session-step", command=...)` dispatch adapter
- [x] Add alias compatibility for legacy `session-*` and `session-step-*` actions
- [x] Add deprecation metadata to legacy alias paths

### Docs/Contract

- [x] Update `mcp/capabilities_manifest.json` to match runtime action shapes
- [x] Update `docs/05-mcp-tool-reference.md` with canonical + compatibility examples

### Tests

- [x] Canonical and legacy action parity tests
- [x] Deprecation metadata presence tests

### Merge Gate

- [x] Both canonical and legacy invocations pass
- [x] Manifest/docs match runtime for action shapes

## P0b Config Foundation: Feature Flags + Capability Truth

### Build

- [x] Add `[feature_flags]` config loading support
- [x] Add env override support for feature flags (document precedence)
- [x] Ensure capability endpoint reports runtime-enabled autonomy flags
- [x] Add startup validation warnings for inconsistent flag states

### Docs/Contract

- [x] Document feature flag config format and env override precedence
- [x] Update capability endpoint documentation

### Convention

- [x] Document "discovery as hints, responses as truth" convention

### Tests

- [x] Feature flag config parsing tests
- [x] Env override precedence tests
- [x] Capability truthfulness tests (including runtime divergence from manifest)
- [x] Startup validation warning tests

### Merge Gate

- [x] Operators can enable/disable flags through documented config paths
- [x] Capability outputs accurate for current runtime

## P1 Supervisor Semantics: Loop Signals

### Build

- [x] Add `loop_signal` field to step response models (step responses only — see Resolved Decision #1)
- [x] Add deterministic mapping per WS3 mapping table (`status`/`pause_reason`/error -> `loop_signal`)
- [x] Add `recommended_actions` for escalation paths
- [x] Add derived `session_signal` summary to `session-status` (if needed)

### Docs/Contract

- [x] Document loop supervisor rule: continue only on `phase_complete`
- [x] Add response examples for all major `loop_signal` variants

### Tests

- [x] Contract tests for each `loop_signal` branch (must cover all rows in WS3 mapping table)
- [x] Regression tests for existing response consumers
- [x] Verify new fields are additive-only (existing consumers unaffected)

### Merge Gate

- [x] Loop supervisor can branch on one field (`loop_signal`)
- [x] No regression in existing response fields

## P2 Observability: Session Monitoring

Depends on: P1 (session_signal derives from loop_signal).

### Build

- [x] Extend session status payload (`last_step_id`, `last_step_type`, `current_task_id`, phase progress, retry counters)
- [x] Add `session-events` action as journal-backed filtered view with pagination (no new persistence — see Resolved Decision #2)
- [x] Standardize event schema for operator tooling
- [x] Validate performance: journal-backed queries under 200ms at design-scale (10 sessions, 10k journal entries each)

### Docs/Contract

- [x] Add operator polling guidance and example queries
- [x] Document event feed schema and pagination rules

### Tests

- [x] Session status field contract tests
- [x] Event feed pagination/cursor tests
- [x] Journal-backed query correctness tests

### Merge Gate

- [x] Operator can monitor loop health from MCP APIs only

## P3 Hardening Completion: Proof + Integrity

### Audit (complete before remediation)

- [x] Produce written gap analysis of current proof/receipt enforcement
- [x] Review gap analysis and confirm remediation scope
- Audit artifact: `_research/proof-receipt-gap-analysis-2026-02-16.md`

### Build

- [x] Remediate gaps identified by audit: tighten consumption/replay semantics
- [x] Close unchecked receipt acceptance paths (field completeness, binding correctness)
- [x] Standardize integrity failure response details

### Scope Exclusion

- [x] Confirmed: signed verification receipts deferred (see Resolved Decision #4)

### Docs/Contract

- [x] Document proof/receipt requirements and failure semantics
- [x] Document receipt construction contract (`command_hash`, `exit_code`, `output_digest`, `issued_at`, `step_id`) for consumer use

### Tests

- [x] Replay and proof-conflict tests
- [x] Verification receipt validation tests (field completeness, binding)
- [x] Receipt construction contract tests (valid and invalid shapes)
- [x] Integrity checksum/audit failure tests

### Merge Gate

- [x] Gap analysis reviewed and remediation scope confirmed
- [x] Existing integrity paths are airtight with no unchecked acceptance
- [x] Receipt construction contract is documented and testable
- [x] Integrity failures deterministic and non-silent

## P4 Posture Profiles

### Build

- [x] Add posture profiles (fixed enumeration: `unattended`, `supervised`, `debug` — not extensible per Resolved Decision #6)
- [x] Map profile defaults to role + bypass + waiver + gate policy
- [x] Add startup validation for unsafe autonomous combinations
- [x] Document and implement role verification preflight pattern for consumer use

### Docs/Contract

- [x] Update sample config with posture examples
- [x] Add operator guidance for profile selection
- [x] Document role verification preflight convention (lightweight call pattern, fail-fast guidance)

### Tests

- [x] Profile parsing/default expansion tests
- [x] Startup validation tests (safe/unsafe)

### Merge Gate

- [x] Unsafe unattended profiles blocked or loudly warned

## P5 Migration Closure + Release Readiness

### Build

- [x] Enable deprecation warnings on legacy action name paths via response envelope metadata (`deprecated.action`, `deprecated.replacement`, `deprecated.removal_target` — see Resolved Decision #5)
- [x] Add server-side `WARN`-level logging for legacy action invocations
- [x] Confirm deprecation timeline: 3 months or 2 minor releases (see Resolved Decision #3)

### Docs/Release

- [x] Update changelog and migration notes with deprecation timeline
- [x] Update runbooks with final supervisor/escalation guidance

### Tests/Validation

- [x] Full autonomy/session-step regression suite green
- [x] Docs/manifest examples validated against runtime behavior
- [x] Deprecation metadata present in legacy action responses

### Merge Gate

- [x] Review checklist complete and release-ready
- [x] Legacy action invocations include machine-readable `deprecated` metadata in response envelope

## P6 Consumer Validation: V2 Skill + End-to-End

Depends on: P0a-P5 (all infrastructure phases landed).

### Build

- [x] Land `foundry-implement-v2` skill definition
- [x] Implement startup preflight (spec resolution, action-shape compatibility, feature flag verification, role/capability verification)
- [x] Implement step-driven execution loop (`session-step-next` -> dispatch -> `session-step-report`)
- [x] Implement step handlers (`implement_task`, `execute_verification` with receipt construction, `run_fidelity_gate`, `address_fidelity_feedback`, `pause`, `complete_spec`)
- [x] Implement deterministic exit behavior for each `loop_signal` category

### Tests

- [x] Preflight tests (disabled flags, denied role, action-shape detection)
- [x] Step loop unit tests (each handler)
- [x] End-to-end integration test: full phase completion against test spec in `unattended` posture
- [x] Escalation exit tests: each non-`phase_complete` signal triggers correct skill exit

### Caveat Coverage

- [x] Caveat A (action-shape): compatibility detection in preflight
- [x] Caveat B (feature flags): fail-fast on disabled features
- [x] Caveat C (role preflight): role verification before session start
- [x] Caveat D (receipt construction): valid receipts in `execute_verification` handler
- [x] Caveat E (discovery vs. runtime): preflight treats responses as truth, not manifest
- [x] Caveat F (step-proof): skill handles proof semantics per P3 contract

### Merge Gate

- [x] V2 skill completes single phase of test spec in `unattended` posture
- [x] Preflight rejects gracefully when prerequisites unmet
- [x] Each `loop_signal` category triggers correct exit behavior
- [x] All six research caveats (A-F) addressed and tested

## Cross-Phase Safety Checks (Run Every PR)

- [x] Envelope contract remains `response-v2`
- [x] Validation and sanitization reviewed for changed inputs
- [x] Error semantics reviewed for actionable remediation
- [x] Feature-flag and authz behavior covered by tests
- [x] Concurrency/timeouts/cancellation behavior reviewed when applicable
- [x] Specs/docs/tests updated together
- [x] New fields are additive-only (rollback safe)

Cross-phase evidence:
1. `response-v2` envelope remains defaulted by shared response helpers and verified by response tests (`src/foundry_mcp/core/responses.py:218`, `tests/test_responses.py:250`).
2. Input validation/sanitization coverage includes posture, feature-flag, and receipt/step payload paths (`tests/unit/test_skills/test_foundry_implement_v2.py:133`, `tests/unit/test_skills/test_foundry_implement_v2.py:163`, `tests/unit/test_core/autonomy/test_handlers_session_step.py:823`).
3. Actionable error semantics are encoded with remediation paths and loop-signal escalation guidance (`src/foundry_mcp/skills/foundry_implement_v2.py:349`, `docs/guides/autonomy-supervisor-runbook.md:60`).
4. Feature-flag + authz behavior is covered by task-shape and preflight tests (`tests/tools/unified/test_task_action_shapes.py:179`, `tests/tools/unified/test_task_action_shapes.py:206`, `tests/unit/test_skills/test_foundry_implement_v2.py:148`).
5. Concurrency/timeouts/cancellation review: no async/concurrency primitives were introduced in this change set (skills preflight/packet shaping + docs/tests only).
6. Specs/docs/tests changed together in this workstream (`skills/foundry-implement-v2/SKILL.md:42`, `docs/guides/autonomy-supervisor-runbook.md:26`, `tests/unit/test_skills/test_foundry_implement_v2.py:524`).
7. Newly added payload data remains additive/optional via `details.recommended_actions` (`src/foundry_mcp/skills/foundry_implement_v2.py:662`).

## Rollback Readiness (Verify Each Phase)

- [x] Confirmed: new response fields do not break existing consumers if removed
- [x] Confirmed: feature flags can disable new behavior without code revert
- [x] Confirmed: tightened validation (P3) has a known relaxation path if false rejections occur
- [x] Confirmed: posture profiles (P4) do not remove direct flag configuration
- [x] Confirmed: v2 skill (P6) is a pure consumer — reverting it has zero impact on MCP server

Rollback evidence:
1. Added fields are optional/additive and isolated to packet details (`src/foundry_mcp/skills/foundry_implement_v2.py:662`).
2. Runtime toggles disable autonomy behavior without code changes (`tests/unit/test_skills/test_foundry_implement_v2.py:133`, `tests/unit/test_config_hierarchy.py:597`).
3. P3 relaxation path is explicitly documented in rollback strategy (`_research/PLAN.md:620`).
4. Direct posture overrides remain supported (with warning) under P4 (`tests/unit/test_config_hierarchy.py:738`).
5. Skill remains consumer-only and is not imported by server runtime paths (`NO_SKILL_IMPORTS_IN_SERVER_RUNTIME_PATHS` verification command).

## Validation Checklist

### Posture Profile Validation

- [x] `unattended` profile: escape hatches closed, `autonomy_runner` role, continues only on `phase_complete`
- [x] `supervised` profile: escape hatches available with reason codes, standard guardrails enforced
- [x] `debug` profile: maximum flexibility, unattended loops rejected
- [x] Supervisor loop dry-run under `unattended`: continues only on `phase_complete`
- [x] Escalation packet generated for each mandatory escalation condition

### Unattended Operation Readiness

- [x] Autonomous identity uses `autonomy_runner`
- [x] Lock bypass disabled
- [x] Gate waiver disabled
- [ ] Alerting configured for integrity and escalation-class failures
- [x] Operator playbook linked and accessible

Alerting verification task (pending ops):
1. Wire alerts for integrity and escalation-class failures using runtime error codes and loop signals (evidence target: alert rule IDs + on-call routing + test alert screenshots/log excerpts).

### End-to-End (after P6)

- [x] V2 skill validated against test spec with `unattended` posture
- [x] Escalation behavior confirmed for each mandatory escalation condition
- [x] Full loop validated: preflight -> session -> steps -> gate -> phase boundary -> stop

## Completion Criteria

- [x] Headless supervisor can safely loop phase-by-phase to spec completion
- [x] Non-success states escalate deterministically
- [x] Operator observability is sufficient without raw state inspection
- [x] Hard boundaries enforced by server policy, not prompt text
- [x] V2 skill completes end-to-end phase execution, validating all infrastructure
- [x] All six research caveats (A-F) addressed and tested

Completion evidence:
1. Phase-by-phase loop behavior verified by repeated `run_single_phase` reaching `spec_complete` (`tests/unit/test_skills/test_foundry_implement_v2.py:387`).
2. Deterministic non-success escalation and packet shaping verified (`tests/unit/test_skills/test_foundry_implement_v2.py:524`, `tests/tools/unified/test_task_action_shapes.py:179`).
3. Observability surfaces validated (`tests/unit/test_core/autonomy/test_handlers_session.py:888`, `tests/unit/test_core/autonomy/test_handlers_session.py:1089`, `docs/guides/autonomy-supervisor-runbook.md:84`).
4. Policy-boundary enforcement validated via authz/feature/posture fail-fast (`tests/unit/test_skills/test_foundry_implement_v2.py:133`, `tests/unit/test_skills/test_foundry_implement_v2.py:163`, `tests/tools/unified/test_server.py:149`).
5. End-to-end unattended phase execution validated (`tests/integration/test_foundry_implement_v2_unattended.py:69`).
6. Caveats A-F are explicitly tracked and checked in this checklist (`_research/PLAN-EXECUTION-CHECKLIST.md:233`).
