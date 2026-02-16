# Autonomy Agent Isolation Guide

Operational guidance for constraining agent-level access during unattended autonomous execution.

Use this guide alongside the [Supervisor Runbook](autonomy-supervisor-runbook.md) and [Configuration Guide](../06-configuration.md).

## Scope

This guide covers the boundary between MCP-level enforcement (server-side, hard) and agent-level access (host-side, soft). It applies when running `foundry-implement-v2` or any autonomous skill under the `autonomy_runner` role with `unattended` posture.

## Enforcement Boundary

### What the MCP server enforces (hard boundaries)

The `autonomy_runner` role restricts MCP tool calls to a 14-action allowlist. Every MCP tool invocation is checked against this list in the common dispatch path (`src/foundry_mcp/tools/unified/common.py`). The server rejects all other actions regardless of agent behavior.

Allowed actions:

```
spec-find, server-capabilities,
session-start, session-resume, session-heartbeat, session-rebase,
session-list, session-status,
session-step-next, session-step-report,
review-fidelity-gate
```

Additionally, the `unattended` posture enforces:

| Control | Value | Effect |
|---|---|---|
| `allow_lock_bypass` | `false` | Server rejects lock bypass requests |
| `allow_gate_waiver` | `false` | Server rejects gate waiver requests |
| `enforce_required_phase_gates` | `true` | Server blocks phase transitions without gate satisfaction |
| `gate_policy` | `strict` | Server validates gate evidence integrity |
| Step proof enforcement | active | Server validates one-time proof tokens |
| Receipt validation | active | Server checks SHA-256 format, UTC timestamps, step binding |
| Spec integrity hash | active | Server detects spec modification at step boundaries |

These controls cannot be weakened by the agent via MCP.

### What the MCP server does NOT enforce (soft boundaries)

The agent runs inside Claude Code, which provides native tools that operate outside MCP authorization:

| Tool | Capability | MCP gated? |
|---|---|---|
| Write / Edit | Modify any file on disk | No |
| Bash | Run arbitrary shell commands | No |
| Read / Glob / Grep | Read any file or directory | No |

The MCP write lock prevents other MCP sessions from claiming the same spec. It does not prevent the agent's native filesystem access.

## Risks Without Isolation

| Risk | Vector | Detection | Prevention |
|---|---|---|---|
| **Spec tampering** | Agent writes spec JSON via Write tool | Detected at next integrity check (hash mismatch → `SPEC_REBASE_REQUIRED`) | Block writes to `specs/` |
| **Config self-escalation** | Agent modifies `foundry-mcp.toml` to change role or posture | Takes effect at next server startup, not current session | Block writes to config files |
| **Unprotocoled file mutation** | Agent modifies source code via Write/Edit outside session-step protocol | Not detected by orchestrator (write lock is MCP-level) | Restrict Write/Edit to expected directories |
| **Unreviewed git operations** | Agent runs `git commit`/`git push` via Bash | Not detected by orchestrator | Restrict git write commands |
| **Audit trail manipulation** | Agent modifies or deletes journal/audit JSON files | Not detected (files are plain JSON on disk) | Make audit directories append-only or out of agent's write path |

## Recommended Mitigations

### 1. Claude Code hooks

Claude Code supports hooks that run before or after tool invocations. Use these to gate agent access during autonomous runs.

Example hook configuration (`~/.claude/hooks.json` or project `.claude/hooks.json`):

```json
{
  "hooks": {
    "pre_tool_use": [
      {
        "tool": "Write",
        "command": "python scripts/guard_autonomous_write.py \"$FILE_PATH\"",
        "description": "Block writes to specs/, config files, and audit directories during autonomous runs"
      },
      {
        "tool": "Edit",
        "command": "python scripts/guard_autonomous_write.py \"$FILE_PATH\"",
        "description": "Block edits to specs/, config files, and audit directories during autonomous runs"
      },
      {
        "tool": "Bash",
        "command": "python scripts/guard_autonomous_bash.py \"$COMMAND\"",
        "description": "Restrict git write operations and config modifications during autonomous runs"
      }
    ]
  }
}
```

Guard logic should block:
- Write/Edit targeting `specs/**/*.json`, `**/foundry-mcp.toml`, `**/.foundry-mcp.toml`, and audit/journal directories.
- Bash commands matching `git commit`, `git push`, `git reset`, `git rebase`, or any command that modifies config files.

Guard logic should allow:
- Write/Edit to source code files (the agent needs this for `implement_task` and `address_fidelity_feedback` steps).
- Bash commands for running tests, linting, and other verification commands (the agent needs this for `execute_verification` steps).
- Read/Glob/Grep operations (read-only, no risk).

### 2. Filesystem sandboxing

For higher assurance, run the agent in a sandboxed environment where:
- `specs/` directory is mounted read-only.
- `foundry-mcp.toml` is mounted read-only.
- Audit/journal directories are outside the agent's writable filesystem or mounted append-only.
- Git credentials are scoped (e.g., read-only deploy key, or no push access).

### 3. Git credential scoping

If the agent must commit during `implement_task` steps:
- Use a Git credential that allows commits to a feature branch only.
- Disable force-push via branch protection rules.
- Require PR review before merge (the agent's commits are reviewable, not directly landed).

If the agent should not commit at all:
- Remove Git write credentials from the agent's environment.
- The skill's SKILL.md does not instruct the agent to commit — commits are an optional step handler behavior.

### 4. Post-session integrity verification

After each autonomous phase run, verify:
- Spec file checksums match the session's recorded `spec_structure_hash`.
- No unexpected config file modifications occurred.
- Journal/audit entries are intact and sequential (no gaps or out-of-order timestamps).
- Git log shows only expected commits.

## Relationship to MCP Hardening

This guide complements, not replaces, the MCP-level hardening. The defense model is layered:

1. **MCP authorization** (server-enforced): Restricts which MCP actions the agent can invoke. Cannot be bypassed by the agent.
2. **Posture policy** (server-enforced): Sets fail-closed defaults for locks, gates, and enforcement. Cannot be weakened by the agent mid-session.
3. **Agent isolation** (caller-enforced, this guide): Restricts the agent's native tool access. Prevents the agent from acting outside the session-step protocol.
4. **Spec integrity** (detective): Hash checks detect spec tampering at step boundaries. Catches tampering that slips past preventive controls.
5. **Audit trail** (detective): Journal and audit entries provide a forensic record. Effective only if audit files are protected from modification.

All five layers should be active for production unattended operation.
