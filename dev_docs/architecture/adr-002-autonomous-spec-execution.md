# ADR-002: Autonomous Spec Execution Control Plane

**Status:** Accepted (Amended)
**Date:** 2026-02-12
**Amended:** 2026-02-16
**Supersedes:** N/A

## Purpose

Define the high-level architecture for autonomous spec execution in Foundry MCP:
- durable session orchestration,
- fidelity-gated phase progression,
- deterministic pause/stop/escalation behavior,
- and server-enforced trust boundaries.

This ADR is intentionally concise. Detailed payload contracts, implementation mapping, and exhaustive test matrices live in companion docs and plans.

## Context

Foundry MCP needs a reliable execution control plane for long-running implementation workflows.

Without a durable control plane, autonomous execution risks:
1. state drift across retries/resumes,
2. ambiguous advancement criteria,
3. silent bypass of quality gates,
4. weak operator visibility,
5. unsafe unattended behavior.

The system must control execution state; it does not generate code itself.

## Decision Summary

Foundry MCP adopts an autonomous execution control plane with these decisions:

1. **Session-Orchestrated Execution**
- Autonomous execution is modeled as a durable session state machine per spec.
- Session state persists across context resets, process restarts, and caller retries.

2. **Lifecycle vs Hot-Path Separation**
- Lifecycle operations and step progression are separate concerns.
- Canonical contract intent is:
  - `task(action="session", command=...)` for lifecycle,
  - `task(action="session-step", command=...)` for orchestration hot path.
- Legacy action aliases may be supported during migration.

3. **Fidelity-Gated Progression**
- Phase advancement is gated by fidelity outcomes.
- Gate evaluation policy is session-configurable (`strict`, `lenient`, `manual`).
- Repeated gate failure is bounded by a cycle cap to prevent spin loops.

4. **One-Phase Primitive Is First-Class**
- `stop_on_phase_completion=true` is a supported operating mode.
- This creates a bounded primitive (`run one phase`) intended for supervisor loops.

5. **Deterministic Stop/Escalation Semantics**
- Non-success conditions must produce explicit, actionable stop states.
- Supervisor behavior should be deterministic:
  - continue only on phase-complete success,
  - escalate on other paused/failed/policy-integrity outcomes.

6. **Server-Enforced Boundaries First**
- Security and policy boundaries are enforced by server logic, not prompt text.
- Feature flags, authorization, gate invariants, and evidence checks are authoritative controls.

7. **Compatibility with Controlled Migration**
- Runtime and discovery contracts must converge.
- If aliases are retained, deprecation behavior must be explicit and documented.

## Architecture Overview

Core components:

1. **Session Lifecycle Manager**
- Start/status/pause/resume/rebase/end/list/reset.
- Owns session identity, status transitions, and recovery entry points.

2. **Step Orchestrator**
- Produces next executable step.
- Requires feedback from prior step before advancing.
- Enforces pause guards, staleness checks, and gate sequencing.

3. **Gate Evaluation Path**
- Fidelity gate actions produce evidence.
- Orchestrator validates and applies policy before phase advancement.

4. **Durable State + Integrity Checks**
- Session state is persisted with versioned schema and migration support.
- Spec-structure integrity is checked to detect drift/mutation requiring reconciliation.

5. **Operator Visibility Surfaces**
- Session/status, progress, and journal/event history provide runtime observability.

## Execution Model (High-Level)

The execution model is a closed loop:
1. Start or resume session.
2. Request next step.
3. Execute step externally (agent/operator).
4. Report step outcome.
5. Orchestrator decides continue/pause/complete/fail.

For phase-bounded operation:
- Run with `stop_on_phase_completion=true`.
- Treat phase-complete pause as successful primitive completion.
- Outer supervisor loops until spec completion or escalation.

## Boundary Hardness Model

Boundaries are classified as:

1. **Hard (server-enforced, fail-closed)**
- feature gating,
- role authorization,
- step identity checks,
- evidence/receipt validation,
- required-gate invariants.

2. **Firm (server-enforced, policy-overridable)**
- write-lock bypass,
- gate waiver.

3. **Soft (instruction/process only)**
- skill prompt rules,
- manual runbook discipline.

Design rule:
- Soft controls are defense-in-depth only.
- Unattended safety depends on hard/firm server policy.

## Posture Model (Operational)

Recommended posture tiers:

1. **strict-prod**
- unattended loop identity with restricted role,
- bypass/waiver disabled,
- continue only on phase-complete success.

2. **staging**
- same guardrails by default,
- controlled tuning and validation.

3. **debug**
- maintainer-operated troubleshooting,
- no unattended autonomous looping.

## Capability and Contract Guidance

1. Discovery and runtime behavior must remain aligned.
2. Canonical contracts should be preferred in docs/examples.
3. If legacy aliases exist, document them as transitional and time-bounded.
4. Response semantics should remain machine-actionable for supervisors and operators.

## Consequences

Positive:
1. Durable, resumable autonomous execution.
2. Deterministic quality-gated progression.
3. Stronger operator control and observability.
4. Better safety posture for unattended loops.

Tradeoffs:
1. More explicit orchestration complexity (state machine + feedback loop).
2. Tighter contracts for callers (must report step outcomes correctly).
3. Need to manage compatibility during contract convergence.

## Out of Scope

This ADR does not define:
1. low-level field-by-field API schemas,
2. file-by-file implementation task lists,
3. exhaustive test case inventory,
4. product-specific skill prompts.

Those belong in tool reference docs, implementation plans, and test suites.

## Amendment Notes (2026-02-16)

This amendment makes ADR-002 concise and architecture-focused by:
1. removing volatile implementation details from the ADR body,
2. clarifying loop-primitive architecture and supervisor semantics,
3. introducing boundary-hardness framing,
4. emphasizing contract/runtime convergence and migration discipline.

## References

- `docs/05-mcp-tool-reference.md`
- `mcp/capabilities_manifest.json`
- `dev_docs/mcp_best_practices/README.md`
- `dev_docs/mcp_best_practices/13-tool-discovery.md`
- `dev_docs/mcp_best_practices/14-feature-flags.md`
- `dev_docs/mcp_best_practices/15-concurrency-patterns.md`
- `dev_docs/mcp_best_practices/08-security-trust-boundaries.md`

