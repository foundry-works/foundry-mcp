# PLAN Execution Checklist

Date: 2026-02-16
Source Plan: `./PLAN.md`

## How to Use

1. Work top-to-bottom by phase (`P0` -> `P5`).
2. Do not start a new phase until the merge gate for current phase is satisfied.
3. Mark each item with `[x]` only after code, docs, and tests are complete.

## Program Readiness

- [ ] Tracking spec/issue created and linked
- [ ] Owners assigned for `P0`-`P5`
- [ ] Rollout environment(s) identified (staging, production)
- [ ] Success metrics agreed (loop success rate, escalation rate, integrity incidents)

## P0 Foundation: Contract + Config Truth

### Build

- [ ] Implement canonical `task(action="session", command=...)`
- [ ] Implement canonical `task(action="session-step", command=...)`
- [ ] Add alias compatibility for legacy `session-*` and `session-step-*` actions
- [ ] Add `[feature_flags]` config loading support
- [ ] Add env override support for feature flags (document precedence)
- [ ] Ensure capability endpoint reports runtime-enabled autonomy flags

### Docs/Contract

- [ ] Update `mcp/capabilities_manifest.json` to match runtime
- [ ] Update `docs/05-mcp-tool-reference.md` with canonical + compatibility examples

### Tests

- [ ] Canonical and legacy action parity tests
- [ ] Feature flag config parsing tests
- [ ] Capability truthfulness tests

### Merge Gate

- [ ] Backward compatibility confirmed
- [ ] Manifest/docs parity confirmed

## P1 Supervisor Semantics: Loop Signals

### Build

- [ ] Add machine-readable `loop_signal` field(s)
- [ ] Add deterministic mapping (`status`/`pause_reason`/error -> `loop_signal`)
- [ ] Add `recommended_actions` for escalation paths

### Docs/Contract

- [ ] Document loop supervisor rule: continue only on `phase_complete`
- [ ] Add response examples for all major `loop_signal` variants

### Tests

- [ ] Contract tests for each `loop_signal` branch
- [ ] Regression tests for existing response consumers

### Merge Gate

- [ ] Loop supervisor can branch on one field (`loop_signal`)

## P2 Observability: Session Monitoring

### Build

- [ ] Extend session status payload (`last_step_id`, `last_step_type`, `current_task_id`, phase progress)
- [ ] Add session events feed (or equivalent) with pagination
- [ ] Standardize event schema for operator tooling

### Docs/Contract

- [ ] Add operator polling guidance and example queries
- [ ] Document event feed schema and pagination rules

### Tests

- [ ] Session status field contract tests
- [ ] Event feed pagination/cursor tests

### Merge Gate

- [ ] Operator can monitor loop health from MCP APIs only

## P3 Hardening Completion: Proof + Integrity

### Build

- [ ] Complete step-proof consumption/replay semantics
- [ ] Tighten verification receipt lifecycle consistency
- [ ] Standardize integrity failure response details

### Docs/Contract

- [ ] Document proof/receipt requirements and failure semantics

### Tests

- [ ] Replay and proof-conflict tests
- [ ] Verification receipt validation tests
- [ ] Integrity checksum/audit failure tests

### Merge Gate

- [ ] Integrity paths deterministic and non-silent

## P4 Posture Profiles

### Build

- [ ] Add posture profiles (`strict-prod`, `staging`, `debug`)
- [ ] Map profile defaults to role + bypass + waiver + gate policy
- [ ] Add startup validation for unsafe autonomous combinations

### Docs/Contract

- [ ] Update sample config with posture examples
- [ ] Add operator guidance for profile selection

### Tests

- [ ] Profile parsing/default expansion tests
- [ ] Startup validation tests (safe/unsafe)

### Merge Gate

- [ ] Unsafe unattended profiles blocked or loudly warned

## P5 Migration Closure + Release Readiness

### Build

- [ ] Add deprecation warnings (if applicable) for legacy action names
- [ ] Prepare final compatibility removal timeline (if applicable)

### Docs/Release

- [ ] Update changelog and migration notes
- [ ] Update runbooks with final supervisor/escalation guidance

### Tests/Validation

- [ ] Full autonomy/session-step regression suite green
- [ ] Docs/manifest examples validated against runtime behavior

### Merge Gate

- [ ] Review checklist complete and release-ready

## Cross-Phase Safety Checks (Run Every PR)

- [ ] Envelope contract remains `response-v2`
- [ ] Validation and sanitization reviewed for changed inputs
- [ ] Error semantics reviewed for actionable remediation
- [ ] Feature-flag and authz behavior covered by tests
- [ ] Concurrency/timeouts/cancellation behavior reviewed when applicable
- [ ] Specs/docs/tests updated together

## Rollout Checklist

### Staging

- [ ] `strict-prod` profile simulated in staging
- [ ] Supervisor loop dry-run: continues only on `phase_complete`
- [ ] Escalation packet generated for each mandatory escalation condition

### Production

- [ ] Autonomous identity uses `autonomy_runner`
- [ ] Lock bypass disabled
- [ ] Gate waiver disabled
- [ ] Alerting configured for integrity and escalation-class failures
- [ ] On-call/operator playbook linked and accessible

## Completion Criteria

- [ ] Headless supervisor can safely loop phase-by-phase to spec completion
- [ ] Non-success states escalate deterministically
- [ ] Operator observability is sufficient without raw state inspection
- [ ] Hard boundaries enforced by server policy, not prompt text

