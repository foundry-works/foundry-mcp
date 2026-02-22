# MCP Autonomy Looping + Hardening Plan

Date: 2026-02-16
Owner: foundry-mcp core
Status: Executed (P0a–P6 landed, post-review hardening complete)

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

---

## 5. Design Rationale

### Why a one-phase primitive?

`run_one_phase(spec_id, workspace)` was chosen as the fundamental unit because it is bounded (single phase max), deterministic (explicit stop at `phase_complete` or terminal status), quality-gated (phase boundary depends on fidelity-gate acceptance), and durable (session state persists for resume/recovery).

The alternative — a full-spec primitive — conflates progress with policy. A single-phase primitive lets the supervisor decide whether to continue, which keeps the escalation boundary clean. The right architecture is two levels: a strict inner primitive (one phase) and a conservative outer supervisor loop that continues only on `phase_complete`.

### Why server policy over prompt discipline?

Skill instructions ("do not bypass write locks") are soft boundaries. The agent can ignore them. For safety-critical controls, the enforcement must come from the server, not the prompt.

The boundary taxonomy:
- **Hard boundaries** — server-enforced, fail-closed. Role allowlists, feature gates, step identity checks, gate evidence validation, required-gate invariants.
- **Firm boundaries** — server-enforced but overridable under policy. Write lock bypass, gate waiver. Converted to effectively hard in unattended posture by disabling the override flags.
- **Soft boundaries** — instructions/process only. Skill prompt rules, human review process. Useful as defense-in-depth, never as primary control.

The design rule: server policy wins over supervisor logic wins over skill instructions. If these disagree, the higher layer prevails.

### Why these specific posture controls?

The threat model assumes five plausible failure modes: confused agent attempting unintended commands, operator accidentally granting too-broad authority, loop thrashing on repeated failures, state/evidence drift across retries, and privileged actor attempting bypass. Host compromise is out of scope.

To address these:
- **Identity separation** — autonomous loop runs as `autonomy_runner`, not `maintainer`. Prevents the loop from invoking escape hatches even if prompted to.
- **Escape hatches disabled** — `allow_lock_bypass=false` and `allow_gate_waiver=false` in unattended posture. Converts firm boundaries to effectively hard.
- **Required gate enforcement** — prevents phase/spec completion when gate obligations are unsatisfied. Without this, a confused agent could skip quality checks.
- **Bounded execution** — `stop_on_phase_completion=true`, session limits on errors/tasks/fidelity cycles. Prevents unbounded thrashing.
- **Deterministic escalation** — only `phase_complete` triggers auto-continue. Every other non-terminal state stops and escalates. This is intentionally conservative: false stops are recoverable, false continues are not.

### Why agent-level guards beyond MCP authorization?

MCP authorization gates MCP tool calls. But the agent also has native tools (Bash, Write, Edit) that operate outside the MCP boundary entirely. The agent could write to spec files, modify config to escalate privileges for future sessions, run `git push` without orchestrator knowledge, or tamper with journal/audit files.

The MCP server detects some of these after the fact (spec integrity hash catches spec modification at the next step boundary), but prevention is better than detection. Hook-based guard scripts (`guard_autonomous_write.py`, `guard_autonomous_bash.py`) add a preventive layer. These are layered mitigations — no single layer is sufficient, but together they raise the bar significantly.

### Why optimistic locking on session mutations?

Session mutations (pause, resume, end, reset, rebase, heartbeat, gate-waiver) load state, check status, then write. Without version checking, a concurrent actor can change status between load and save, and the second writer silently overwrites. The `state_version` field was already being incremented but never verified on save. Adding `expected_version` to `save()` closes this gap with minimal API change and full backward compatibility (omitting the parameter skips the check).

### Why bound the proof store?

Long-running sessions accumulate proof records proportional to lifetime. Without cleanup, a session with thousands of steps retains thousands of proof records indefinitely. TTL eviction (1h) removes stale records; LRU cap (500) provides a hard upper bound. These limits are generous for normal operation but prevent unbounded growth in edge cases.

### Why fail rebase when backup is missing?

When the backup spec for structural diff computation is missing, the code previously created an empty diff — meaning `removed_completed_tasks` was always empty. This silently lost task completion history. The guard makes this failure explicit: if completed tasks exist and the backup is gone, the rebase fails unless forced. This preserves the principle that data loss should never be silent.

### Continue vs. escalate: the decision matrix

The supervisor's continue/stop decision is the most safety-critical logic in the system. The principle: only continue on unambiguous success.

| Signal | Continue? | Why |
|---|---|---|
| `phase_complete` | Yes | Unambiguous single-phase success |
| `spec_complete` | No (done) | Overall spec finished |
| `fidelity_cycle_limit` | Escalate | Anti-spin safety stop; auto-continuing defeats the purpose |
| `gate_failed` / `gate_review_required` | Escalate | Quality boundary not met; requires human judgment |
| `blocked` / `error_threshold` | Escalate | Dependencies or repeated failures need investigation |
| `context_limit` / `heartbeat_stale` / `step_stale` | Escalate | Infrastructure/health issue |
| Integrity errors (`GATE_AUDIT_FAILURE`, `GATE_INTEGRITY_CHECKSUM`) | Escalate (high) | Potential tamper; freeze and audit |
| `FEATURE_DISABLED` / `AUTHORIZATION` | Escalate | Environment misconfiguration |

---

## 6. Workstreams

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

## 7. Resolved Decisions

1. **`loop_signal` placement:** Step responses only. `session-status` may carry a derived summary, but the authoritative field lives on step responses.
2. **`session-events` implementation:** Journal-backed filtered view, not new persistence.
3. **Deprecation window:** 3 months or 2 minor releases, whichever comes later.
4. **Signed verification receipts:** Deferred. Current threat model excludes host compromise.
5. **Deprecation warning mechanism:** Machine-readable metadata in response envelope, not just server-side logs.
6. **Posture profile extensibility:** Fixed enumeration only. Custom behavior uses direct flag configuration.
7. **Phase dependencies:** P1 depends on P0a+P0b. P2 depends on P1. P6 depends on P0a-P5.

## 8. Research Caveats (A–F)

These were identified during branch analysis and drove preflight/compatibility design in the v2 skill.

- **A. Action-shape mismatch.** Discovery/manifest indicated `task action="session"` but runtime exposed concrete names like `session-start`. Skill uses runtime compatibility detection at startup.
- **B. Feature flags are fail-closed.** Session and fidelity-gate handlers reject when flags are disabled. Skill preflights and fails fast with actionable remediation.
- **C. Role requirements can block the flow.** Authorization denies actions outside role allowlists. Skill verifies role/capability during preflight.
- **D. Verification receipt requirement is strict.** `execute_verification` success reports require well-formed `verification_receipt`. Skill constructs receipts with all required fields (command_hash, exit_code, output_digest, issued_at, step_id).
- **E. Capability metadata may not represent runtime state.** Server capability surfaces can be static descriptors. Skill treats discovery as hints and tool responses as runtime truth.
- **F. Step-proof enforcement.** Proof record plumbing exists in persistence; orchestration relies on step identity + replay cache. Post-review hardening (C3) added bounds and cleanup.

## 9. Acceptance Criteria (Program-Level)

1. A headless supervisor can safely loop on `phase_complete` and stop/escalate otherwise.
2. Runtime capability/feature outputs are truthful.
3. Operators can observe progress and diagnose stop conditions from MCP APIs alone.
4. Escape hatches are controlled by explicit posture policy, not prompt discipline.
5. All new/changed contracts are documented and test-covered.
6. The `foundry-implement-v2` skill can complete a single phase end-to-end in `unattended` posture.
7. All six research caveats (A–F) are addressed and tested.

## 10. Post-Review Hardening (2026-02-16)

Senior engineering review of the landed branch identified 20 remediation tasks across 5 categories. All implemented and tested same day.

### Critical — Concurrency & Data Safety (C1–C3)

- Optimistic locking on all 7 session mutation sites with `VersionConflictError`.
- Rebase backup guard: fails with `REBASE_BACKUP_MISSING` when completed tasks are at risk.
- Proof store bounds: TTL eviction (1h) + LRU cap (500 records).

### Security — Agent-Level Soft Boundary Hardening (S1–S5)

- Guard scripts: `scripts/guard_autonomous_write.py` and `scripts/guard_autonomous_bash.py`.
- Docs: `docs/guides/autonomy-agent-isolation.md`, SKILL.md agent isolation section, supervisor runbook isolation preflight.
- Env var controls: `FOUNDRY_GUARD_DISABLED`, `FOUNDRY_GUARD_ALLOW_GIT_COMMIT`, `FOUNDRY_GUARD_EXTRA_BLOCKED`, `FOUNDRY_GUARD_EXTRA_ALLOWED`.

### High — Observability & Maintainability (H1–H4)

- Loop signal consolidated to single `attach_loop_metadata()` attachment point.
- Journal write observability via `meta.audit_status` on all session responses.
- Handler file splitting: `handlers_session.py` (2,758→5 files) and `orchestrator.py` (2,255→2 files).
- Config provenance logging for all security-relevant settings.

### Medium — Model Hardening (M1–M4)

- Cross-field Pydantic validators on session models.
- Deprecation enforcement with hard error past removal target.
- `reason_detail` bounded to 2,000 chars.
- Authorization denial audit logging.

### Test Coverage (T1–T5)

- Loop signal exhaustiveness (36 parametrized cases).
- Step proof expiration with time advancement.
- Verification receipt timing boundaries.
- GC-by-TTL for terminal sessions (9 parametrized cases).
- Config env var override provenance tests.
