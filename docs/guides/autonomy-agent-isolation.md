# Autonomy Agent Isolation Guide

Operational guidance for constraining agent-level access during unattended autonomous execution.

Use this guide alongside the [Supervisor Runbook](autonomy-supervisor-runbook.md) and [Configuration Guide](../06-configuration.md).

## Scope

This guide covers the boundary between MCP-level enforcement (server-side, hard) and agent-level access (host-side, soft). It applies when running `foundry-implement-v2` or any autonomous skill under the `autonomy_runner` role with `unattended` posture.

MCP authorization gates MCP tool calls only, not native Claude Code tools. The caller is responsible for constraining the agent's native tool access using the mechanisms described below. Without these constraints, the agent can bypass MCP enforcement via filesystem writes, shell commands, and git operations.

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

## Threat Model

| Risk | Vector | Detection | Prevention |
|---|---|---|---|
| **Spec tampering** | Agent writes spec JSON via Write tool | Detected at next integrity check (hash mismatch -> `SPEC_REBASE_REQUIRED`) | Block writes to `specs/` via hook + pre-commit guard |
| **Config self-escalation** | Agent modifies `foundry-mcp.toml` to change role or posture | Takes effect at next server startup, not current session | Block writes to config files via hook + read-only mount |
| **Unprotocoled file mutation** | Agent modifies source code via Write/Edit outside session-step protocol | Not detected by orchestrator (write lock is MCP-level) | Restrict Write/Edit to expected directories |
| **Unreviewed git operations** | Agent runs `git commit`/`git push` via Bash | Not detected by orchestrator | Restrict git write commands via hook |
| **Audit trail manipulation** | Agent modifies or deletes journal/audit JSON files | Not detected (files are plain JSON on disk) | Block writes to audit dirs via hook + append-only storage |

## Guard Scripts

Two guard scripts implement the hook logic for all five protection domains. These are the reference implementation for Claude Code hooks during autonomous execution.

### `scripts/guard_autonomous_write.py`

Blocks Write/Edit operations targeting protected paths. Covers spec files, config files, session state, and audit/journal directories.

**Protected paths:**

| Pattern | Protection domain |
|---|---|
| `specs/**/*.json` | Spec file integrity |
| `**/foundry-mcp.toml` | Config self-escalation |
| `**/.foundry-mcp.toml` | Config self-escalation |
| `.foundry-mcp/sessions/` | Session state tampering |
| `.foundry-mcp/journals/` | Audit trail integrity |
| `.foundry-mcp/audit/` | Audit trail integrity |
| `.foundry-mcp/proofs/` | Proof record integrity |

**Environment variables:**

| Variable | Effect |
|---|---|
| `FOUNDRY_GUARD_DISABLED=1` | Bypass all checks (emergency escape hatch) |
| `FOUNDRY_GUARD_EXTRA_BLOCKED` | Colon-separated additional path prefixes to block |
| `FOUNDRY_GUARD_EXTRA_ALLOWED` | Colon-separated path prefixes to allow (evaluated before block rules) |

### `scripts/guard_autonomous_bash.py`

Restricts Bash commands during autonomous runs. Blocks git write operations and shell commands targeting protected files.

**Blocked operations:**
- `git commit`, `git push`, `git reset`, `git rebase`, `git checkout`, `git clean`, `git merge`, `git cherry-pick`, `git revert`
- Shell redirections/pipes writing to spec or config files (`> specs/*.json`, `tee foundry-mcp.toml`)
- `rm` targeting audit/journal directories
- `sed -i` / `awk -i` on protected files

**Allowed operations:**
- `git status`, `git diff`, `git log`, `git show`, `git branch` (read-only)
- `pytest`, `python -m pytest`, `make test`, `npm test` (testing/verification)
- General shell commands for inspection and verification

**Environment variables:**

| Variable | Effect |
|---|---|
| `FOUNDRY_GUARD_DISABLED=1` | Bypass all checks (emergency escape hatch) |
| `FOUNDRY_GUARD_ALLOW_GIT_COMMIT=1` | Allow `git commit` during implementation steps |

## Hook Configuration

### Claude Code hooks

Configure hooks in the project's `.claude/settings.json` or `~/.claude/settings.json`:

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

The hooks run before each tool invocation. A non-zero exit code from the guard script blocks the tool call and surfaces the error message to the agent.

### Step-aware git commit allowance

If the autonomous skill legitimately needs to commit during `implement_task` or `address_fidelity_feedback` steps, the supervisor should set `FOUNDRY_GUARD_ALLOW_GIT_COMMIT=1` in the agent's environment. This allows `git commit` while still blocking `git push`, `git reset`, and other destructive operations.

The recommended approach:
1. The supervisor sets `FOUNDRY_GUARD_ALLOW_GIT_COMMIT=1` before launching the agent.
2. The agent commits implementation work via `git commit`.
3. `git push` remains blocked — the supervisor handles pushing after review.
4. After the phase completes, the supervisor reviews commits before pushing.

If the agent should not commit at all, omit `FOUNDRY_GUARD_ALLOW_GIT_COMMIT` (default: blocked).

## Protection Domains

### Spec file integrity

**Risk:** The agent can modify spec JSON files directly via the Write tool, bypassing the session protocol. The orchestrator detects this at the next spec integrity check (hash mismatch -> `SPEC_REBASE_REQUIRED`), but the agent could commit the tampered spec to git before the check fires.

**Mitigations (layered):**

1. **Hook guard (preventive):** `guard_autonomous_write.py` blocks Write/Edit to `specs/**/*.json`.

2. **Pre-commit git hook (preventive):** Add a git pre-commit hook that validates spec file checksums against the session's recorded hash. Example:

   ```bash
   #!/bin/bash
   # .git/hooks/pre-commit — block spec file modifications during autonomous runs
   if [ -n "$FOUNDRY_AUTONOMOUS_SESSION" ]; then
     staged_specs=$(git diff --cached --name-only -- 'specs/*.json' 'specs/**/*.json')
     if [ -n "$staged_specs" ]; then
       echo "ERROR: Spec file modifications are not allowed during autonomous sessions."
       echo "Modified specs: $staged_specs"
       echo "Use the MCP session-rebase protocol to update specs."
       exit 1
     fi
   fi
   ```

   Set `FOUNDRY_AUTONOMOUS_SESSION=1` in the agent's environment to activate.

3. **MCP integrity check (detective):** The orchestrator computes `spec_structure_hash` at session start and verifies it at each step boundary. A mismatch emits `SPEC_REBASE_REQUIRED` and pauses the session.

4. **Filesystem mount (preventive, high assurance):** Mount `specs/` read-only in the agent's container or sandbox.

### Config file protection

**Risk:** The agent can write to `foundry-mcp.toml` to change `role = "maintainer"` or `profile = "debug"`. The MCP server reads config at startup, so this would not affect the current session, but it would escalate privileges for future sessions.

**Mitigations (layered):**

1. **Hook guard (preventive):** `guard_autonomous_write.py` blocks Write/Edit to `**/foundry-mcp.toml` and `**/.foundry-mcp.toml`.

2. **Bash guard (preventive):** `guard_autonomous_bash.py` blocks shell redirections and `sed -i` targeting config files.

3. **Read-only mount (preventive, high assurance):** Mount `foundry-mcp.toml` read-only.

4. **Startup integrity check (detective):** If running in `unattended` posture, check whether the config file was modified since the last server startup. Log a prominent warning if so. Implementation: record config file mtime and checksum at startup in server state; compare at next startup.

   The supervisor can implement this check:

   ```bash
   # Before launching agent, record config checksum
   CONFIG_HASH=$(sha256sum foundry-mcp.toml | cut -d' ' -f1)

   # ... run autonomous phase ...

   # After phase completes, verify config integrity
   POST_HASH=$(sha256sum foundry-mcp.toml | cut -d' ' -f1)
   if [ "$CONFIG_HASH" != "$POST_HASH" ]; then
     echo "WARNING: foundry-mcp.toml was modified during autonomous execution"
     echo "  before: $CONFIG_HASH"
     echo "  after:  $POST_HASH"
     # Restore from backup or investigate
   fi
   ```

### Git operation guarding

**Risk:** The agent can run `git commit`, `git push`, or `git reset` via Bash without orchestrator knowledge or approval. Unreviewed code could be pushed to remote. The agent could commit tampered specs, skip gate results, or force-push over reviewed work.

**Mitigations (layered):**

1. **Bash guard (preventive):** `guard_autonomous_bash.py` blocks all git write operations by default. Only read-only git commands are allowed.

2. **Selective commit allowance:** Set `FOUNDRY_GUARD_ALLOW_GIT_COMMIT=1` to permit `git commit` during implementation steps. `git push` remains blocked — the supervisor pushes after review.

3. **Git credential scoping (preventive):**
   - Use a credential that allows commits to a feature branch only.
   - Disable force-push via branch protection rules.
   - Require PR review before merge.
   - If the agent should not commit at all, remove Git write credentials from the environment.

4. **Pre-commit git hook (detective):** The spec integrity pre-commit hook (above) prevents committing spec modifications. Additional hooks can enforce commit message format or block commits to protected branches.

5. **Post-phase review:** The supervisor reviews `git log` after each phase to verify only expected commits were created.

### Audit file protection

**Risk:** Journal and audit ledger files are on-disk JSON. The agent can modify or delete them via Write/Bash, compromising the audit trail.

**Mitigations (layered):**

1. **Hook guard (preventive):** `guard_autonomous_write.py` blocks Write/Edit to `.foundry-mcp/journals/`, `.foundry-mcp/audit/`, and `.foundry-mcp/proofs/`.

2. **Bash guard (preventive):** `guard_autonomous_bash.py` blocks `rm` commands targeting audit directories.

3. **Separate storage (preventive, high assurance):** Run the MCP server with audit files in a directory the agent cannot write to. Configure the MCP server's storage path to a location outside the agent's writable workspace:

   ```toml
   [storage]
   base_path = "/var/foundry-mcp/data"  # Outside agent workspace
   ```

4. **Append-only filesystem (preventive, high assurance):** Use filesystem-level append-only attributes:

   ```bash
   # Linux: set append-only attribute on audit directory
   chattr +a /var/foundry-mcp/data/journals/
   chattr +a /var/foundry-mcp/data/audit/
   ```

5. **Integrity verification (detective):** After each phase, verify journal entries are intact:
   - Check for sequential timestamps (no gaps or reordering).
   - Verify entry count matches expected operations.
   - Compare journal event IDs against session event stream.

   The supervisor can verify journal integrity:

   ```bash
   # Compare MCP event count with on-disk journal entries
   # MCP provides authoritative event count via session-events API
   ```

## Filesystem Sandboxing (High Assurance)

For production unattended operation with maximum isolation, run the agent in a sandboxed environment:

| Path | Mount | Rationale |
|---|---|---|
| `specs/` | read-only | Prevents spec tampering |
| `foundry-mcp.toml` | read-only | Prevents config self-escalation |
| `.foundry-mcp/journals/` | read-only or separate volume | Prevents audit manipulation |
| `.foundry-mcp/audit/` | read-only or separate volume | Prevents audit manipulation |
| `.foundry-mcp/proofs/` | read-only or separate volume | Prevents proof manipulation |
| `src/`, `tests/`, etc. | read-write | Agent needs write access for implementation |

Docker example:

```bash
docker run \
  -v $(pwd)/specs:/workspace/specs:ro \
  -v $(pwd)/foundry-mcp.toml:/workspace/foundry-mcp.toml:ro \
  -v /var/foundry-mcp/data:/data:rw \
  -e FOUNDRY_MCP_ROLE=autonomy_runner \
  -e FOUNDRY_MCP_AUTONOMY_POSTURE=unattended \
  -e FOUNDRY_AUTONOMOUS_SESSION=1 \
  -e FOUNDRY_GUARD_ALLOW_GIT_COMMIT=1 \
  foundry-agent:latest
```

## Post-Session Integrity Verification

After each autonomous phase, the supervisor should verify:

1. **Spec integrity:** Spec file checksums match the session's recorded `spec_structure_hash`.
2. **Config integrity:** `foundry-mcp.toml` checksum matches the pre-session value.
3. **Audit completeness:** Journal entries are sequential with no gaps or out-of-order timestamps.
4. **Git history:** `git log` shows only expected commits (implementation work, no spec or config changes).
5. **Proof store:** No unexpected proof records or missing consumed proofs.

Supervisor verification script:

```bash
#!/bin/bash
# post_phase_verify.sh — run after each autonomous phase
set -euo pipefail

SESSION_ID="$1"
PRE_CONFIG_HASH="$2"

echo "=== Post-Phase Integrity Verification ==="

# 1. Config file integrity
POST_CONFIG_HASH=$(sha256sum foundry-mcp.toml | cut -d' ' -f1)
if [ "$PRE_CONFIG_HASH" != "$POST_CONFIG_HASH" ]; then
  echo "FAIL: foundry-mcp.toml modified during session"
  exit 1
fi
echo "PASS: Config file unchanged"

# 2. Spec files in git (no uncommitted spec changes)
SPEC_CHANGES=$(git diff --name-only -- 'specs/*.json' 'specs/**/*.json' 2>/dev/null)
if [ -n "$SPEC_CHANGES" ]; then
  echo "FAIL: Uncommitted spec file changes detected: $SPEC_CHANGES"
  exit 1
fi
echo "PASS: No uncommitted spec changes"

# 3. Git log review (no spec/config commits)
SUSPECT_COMMITS=$(git log --oneline --diff-filter=M -- 'specs/*.json' 'foundry-mcp.toml' 2>/dev/null | head -5)
if [ -n "$SUSPECT_COMMITS" ]; then
  echo "WARN: Commits touching spec/config files: $SUSPECT_COMMITS"
fi

echo "=== Verification complete ==="
```

## Relationship to MCP Hardening

This guide complements, not replaces, the MCP-level hardening. The defense model is layered:

1. **MCP authorization** (server-enforced): Restricts which MCP actions the agent can invoke. Cannot be bypassed by the agent.
2. **Posture policy** (server-enforced): Sets fail-closed defaults for locks, gates, and enforcement. Cannot be weakened by the agent mid-session.
3. **Agent isolation** (caller-enforced, this guide): Restricts the agent's native tool access. Prevents the agent from acting outside the session-step protocol.
4. **Spec integrity** (detective): Hash checks detect spec tampering at step boundaries. Catches tampering that slips past preventive controls.
5. **Audit trail** (detective): Journal and audit entries provide a forensic record. Effective only if audit files are protected from modification.

All five layers should be active for production unattended operation.

## Quick Reference

| Task | Hook guard | Filesystem | Git hook | Post-session |
|---|---|---|---|---|
| Spec file integrity | `guard_autonomous_write.py` blocks `specs/**/*.json` | Mount `specs/` read-only | Pre-commit blocks spec changes | Verify `spec_structure_hash` |
| Config protection | `guard_autonomous_write.py` blocks `**/foundry-mcp.toml` | Mount config read-only | N/A | Compare config checksum |
| Git operation guarding | `guard_autonomous_bash.py` blocks git write ops | Scope git credentials | N/A | Review `git log` |
| Audit protection | `guard_autonomous_write.py` blocks `.foundry-mcp/journals\|audit\|proofs/` | Separate volume or append-only | N/A | Verify sequential timestamps |
| Session state protection | `guard_autonomous_write.py` blocks `.foundry-mcp/sessions/` | MCP server manages via protocol | N/A | N/A (MCP-enforced) |
