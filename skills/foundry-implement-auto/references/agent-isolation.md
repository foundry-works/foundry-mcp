# Agent Isolation Constraints

The MCP server enforces authorization for MCP tool calls only. Native Claude Code tools (Write, Edit, Bash) operate outside MCP authorization. The following constraints are enforced by the caller via Claude Code hooks and environment configuration. Violations are blocked by the guard scripts before the tool executes.

## Contents

- [Filesystem Restrictions](#filesystem-restrictions)
- [Shell Restrictions](#shell-restrictions)
- [Allowed Operations](#allowed-operations)

## Filesystem Restrictions

Enforced by `scripts/guard_autonomous_write.py`:

| Path | Reason |
|------|--------|
| `specs/**/*.json` | Use the MCP session-rebase protocol for spec changes |
| `foundry-mcp.toml`, `.foundry-mcp.toml` | Config is read-only during autonomous execution |
| `.foundry-mcp/sessions/` | MCP server manages via protocol |
| `.foundry-mcp/journals/` | MCP server manages via protocol |
| `.foundry-mcp/audit/` | MCP server manages via protocol |
| `.foundry-mcp/proofs/` | MCP server manages via protocol |

## Shell Restrictions

Enforced by `scripts/guard_autonomous_bash.py`:

| Prohibited | Notes |
|-----------|-------|
| `git push`, `git reset`, `git rebase` | Destructive remote/history operations |
| `git checkout`, `git clean`, `git merge` | Destructive working tree operations |
| `git commit` | Allowed only when `FOUNDRY_GUARD_ALLOW_GIT_COMMIT=1` is set by supervisor |
| Shell redirections to spec/config/audit files | Bypass prevention |

## Allowed Operations

| Operation | Scope |
|-----------|-------|
| Write/Edit source files | `src/`, `tests/`, etc. for `implement_task` and `address_fidelity_feedback` steps |
| Bash for tests and linting | During `execute_verification` steps |
| Read/Glob/Grep | Unrestricted (read-only access) |
| `git status`, `git diff`, `git log`, `git show` | Read-only git operations |

For full details on isolation architecture, guard scripts, and high-assurance options, see `docs/guides/autonomy-agent-isolation.md`.
