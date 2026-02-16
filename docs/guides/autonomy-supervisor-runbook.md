# Autonomy Supervisor Runbook

Operational guidance for supervising autonomous session-step loops in production.

Use this runbook with the [MCP Tool Reference](../05-mcp-tool-reference.md) and [Configuration](../06-configuration.md).

## Scope

This runbook covers:

- startup preflight checks before autonomous session start
- unattended supervisor loop behavior
- deterministic escalation handling by `loop_signal`
- operator polling and event-feed usage
- release migration reminders for legacy action names

## Preflight Checklist

Run these checks before `task(action="session", command="start", ...)`:

1. Resolve and verify target spec:
   - `spec(action="find", spec_id=...)`
2. Verify runtime autonomy flags:
   - `server(action="capabilities")`
   - Require `data.runtime.autonomy.enabled_now.autonomy_sessions == true`
   - Reject unattended runs when `data.runtime.autonomy.posture_profile == "debug"`
3. Verify role authorization early:
   - Preferred: `task(action="session", command="list", limit=1)`
   - Legacy fallback: `task(action="session-list", limit=1)`
4. If preflight returns `AUTHORIZATION` or `FEATURE_DISABLED`, stop and escalate with remediation.

## Agent Isolation Preflight

Before launching an autonomous agent, verify the caller-side isolation controls are active. MCP authorization gates MCP tools only — native Claude Code tools (Write, Edit, Bash) require separate constraints.

**Required:**

1. **Hook guards installed.** Verify Claude Code hooks are configured in `.claude/settings.json`:
   - `guard_autonomous_write.py` registered for Write and Edit tools.
   - `guard_autonomous_bash.py` registered for Bash tool.
   - Test hooks with a dry run before first autonomous session.

2. **Environment variables set:**
   - `FOUNDRY_AUTONOMOUS_SESSION=1` — activates git pre-commit guards.
   - `FOUNDRY_GUARD_ALLOW_GIT_COMMIT=1` — set only if the agent should be able to commit (omit to block all git writes).

3. **Config checksum recorded.** Record `sha256sum foundry-mcp.toml` before launching the agent. Compare after phase completes.

**Recommended (high assurance):**

4. **Filesystem mounts.** Mount `specs/` and `foundry-mcp.toml` read-only if running in a container.
5. **Git credential scoping.** Use a credential limited to feature branch commits. Disable force-push via branch protection.
6. **Audit storage separation.** Place `.foundry-mcp/journals/` and `.foundry-mcp/audit/` on a separate volume or set append-only attributes.

For full details, see [Agent Isolation Guide](autonomy-agent-isolation.md).

## Post-Phase Integrity Verification

After each autonomous phase completes (any `loop_signal`), verify:

1. **Config integrity:** `sha256sum foundry-mcp.toml` matches pre-session value.
2. **Spec integrity:** No uncommitted changes to `specs/**/*.json`. No unexpected commits touching spec files.
3. **Git history:** `git log` shows only implementation commits. No spec, config, or audit file modifications.
4. **Audit completeness:** Journal entries are sequential with no timestamp gaps.

See the `post_phase_verify.sh` script in the [Agent Isolation Guide](autonomy-agent-isolation.md#post-session-integrity-verification).

## Start Session

Use canonical action shape:

```json
{
  "action": "session",
  "command": "start",
  "spec_id": "my-feature-spec-001",
  "gate_policy": "strict",
  "stop_on_phase_completion": true,
  "auto_retry_fidelity_gate": true,
  "enforce_autonomy_write_lock": true
}
```

## Supervisor Loop Contract

Loop sequence:

1. `task(action="session-step", command="next", session_id=...)`
2. Read `data.loop_signal`
3. Continue only when `loop_signal == "phase_complete"` and a subsequent phase run is intended
4. For all other non-null signals, stop and escalate
5. Execute step work only when `data.next_step` is present
6. Report results with `task(action="session-step", command="report", ...)`

## `loop_signal` Escalation Matrix

| loop_signal | Operator action |
|---|---|
| `phase_complete` | Successful phase stop. Queue next phase explicitly if needed. |
| `spec_complete` | Successful terminal stop. Close session/supervisor run. |
| `paused_needs_attention` | Stop and route to human review (gate findings, policy pause, stale state, or limits). |
| `failed` | Stop and investigate failure context before retry/rebase. |
| `blocked_runtime` | Stop immediately. Resolve authorization, feature flags, or integrity prerequisites first. |

## Escalation Packet

For every non-continue outcome, record:

- `spec_id`
- `session_id`
- `loop_signal`
- `final_status`
- `pause_reason` (if present)
- `last_step_id`
- `active_phase_id`
- `recommended_actions` (if present)

## Monitoring and Polling

Use MCP APIs only (no raw state file inspection required):

- Poll `task(action="session", command="status", session_id=...)` every 10-30 seconds.
- Read `last_step_id`, `last_step_type`, `current_task_id`, `active_phase_progress`, and `retry_counters`.
- Poll `task(action="session-events", session_id=..., cursor=..., limit=...)` for timeline updates.

## Legacy Action Migration

Legacy action names (for example `session-start`, `session-step-next`) are still accepted during migration.

- Legacy responses include:
  - `meta.deprecated.action`
  - `meta.deprecated.replacement`
  - `meta.deprecated.removal_target`
- Server logs emit `WARN` entries for legacy invocation paths.
- Removal target is **3 months or 2 minor releases (whichever is later)**.
