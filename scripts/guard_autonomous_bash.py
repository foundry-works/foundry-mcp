#!/usr/bin/env python3
"""Guard script for Claude Code hooks: restricts Bash commands during autonomous runs.

Usage (in Claude Code hook configuration):
    command: python scripts/guard_autonomous_bash.py "$COMMAND"

Exit codes:
    0 — command allowed
    1 — command blocked (prints reason to stderr)

Blocked commands:
    - git commit, git push, git reset, git rebase, git checkout (write ops)
    - git clean, git stash drop (destructive ops)
    - Direct writes to protected config/spec/audit files via shell

Allowed commands:
    - git status, git diff, git log, git show, git branch (read-only git)
    - pytest, python -m pytest, make test, npm test (testing)
    - python, node, bash -c (general execution for verification)
    - cat, head, tail, grep, find, ls, wc (read-only inspection)

Environment variables:
    FOUNDRY_GUARD_DISABLED=1        — bypass all checks (emergency escape hatch)
    FOUNDRY_GUARD_ALLOW_GIT_COMMIT=1 — allow git commit (for implement_task steps)
"""
import os
import re
import sys


# Git subcommands that are write operations (blocked by default)
_GIT_WRITE_SUBCOMMANDS = {
    "commit",
    "push",
    "reset",
    "rebase",
    "checkout",
    "clean",
    "merge",
    "cherry-pick",
    "revert",
    "stash",
    "tag",
    "branch -d",
    "branch -D",
    "branch -m",
    "branch -M",
    "remote add",
    "remote remove",
    "remote set-url",
}

# Git subcommands that are read-only (always allowed)
_GIT_READ_SUBCOMMANDS = {
    "status",
    "diff",
    "log",
    "show",
    "branch",
    "remote",
    "describe",
    "rev-parse",
    "ls-files",
    "ls-tree",
    "cat-file",
    "reflog",
    "shortlog",
    "blame",
    "bisect",
    "grep",
    "stash list",
    "stash show",
    "config --get",
    "config --list",
    "config -l",
}

# Patterns for config/spec file writes via shell commands
_SHELL_WRITE_PATTERNS = [
    # Direct writes to config files
    re.compile(r"(?:>|>>|tee)\s+.*foundry-mcp\.toml"),
    re.compile(r"(?:>|>>|tee)\s+.*\.foundry-mcp\.toml"),
    # Direct writes to spec files
    re.compile(r"(?:>|>>|tee)\s+.*specs/.*\.json"),
    # cp/mv to protected locations
    re.compile(r"(?:cp|mv)\s+.*(?:foundry-mcp\.toml|specs/.*\.json)"),
    # rm of audit/journal files
    re.compile(r"rm\s+.*\.foundry-mcp/(?:journals|audit|proofs|sessions)/"),
    # sed/awk in-place edits of protected files
    re.compile(r"sed\s+-i.*(?:foundry-mcp\.toml|specs/.*\.json)"),
    re.compile(r"awk\s+-i.*(?:foundry-mcp\.toml|specs/.*\.json)"),
    # chmod/chown of protected paths
    re.compile(r"(?:chmod|chown)\s+.*\.foundry-mcp/"),
]


def _extract_git_subcommand(command: str) -> str | None:
    """Extract the git subcommand from a command string."""
    # Match 'git <subcommand>' patterns, handling flags before subcommand
    match = re.search(r"\bgit\s+(?:-\w+\s+)*(\S+(?:\s+-[dDmM])?(?:\s+\S+)?)", command)
    if match:
        return match.group(1).strip()
    return None


def _is_git_read_only(subcommand: str) -> bool:
    """Check if a git subcommand is read-only."""
    for read_cmd in _GIT_READ_SUBCOMMANDS:
        if subcommand.startswith(read_cmd):
            return True
    return False


def _is_git_write(subcommand: str) -> bool:
    """Check if a git subcommand is a write operation."""
    for write_cmd in _GIT_WRITE_SUBCOMMANDS:
        if subcommand.startswith(write_cmd):
            return True
    return False


def check_command(command: str) -> tuple[bool, str]:
    """Check whether a bash command is allowed.

    Returns:
        (allowed, reason) — allowed=True means command is permitted.
    """
    if os.environ.get("FOUNDRY_GUARD_DISABLED") == "1":
        return True, "guard disabled via FOUNDRY_GUARD_DISABLED=1"

    # Check for git commands
    if re.search(r"\bgit\b", command):
        subcommand = _extract_git_subcommand(command)

        if subcommand is None:
            # Bare 'git' or unrecognized — allow (e.g., 'git --version')
            return True, "bare git command allowed"

        # Check read-only first
        if _is_git_read_only(subcommand):
            return True, f"git read-only operation: {subcommand}"

        # Check write operations
        if _is_git_write(subcommand):
            # Special case: allow git commit when explicitly permitted
            if subcommand.startswith("commit") and os.environ.get("FOUNDRY_GUARD_ALLOW_GIT_COMMIT") == "1":
                return True, "git commit allowed via FOUNDRY_GUARD_ALLOW_GIT_COMMIT=1"

            return False, f"blocked: git write operation '{subcommand}' is not allowed during autonomous execution"

        # Unknown git subcommand — allow with caution (fail-open for unknown read ops)
        return True, f"unknown git subcommand '{subcommand}' — allowed (not in block list)"

    # Check for shell write patterns targeting protected files
    for pattern in _SHELL_WRITE_PATTERNS:
        if pattern.search(command):
            return False, f"blocked: shell command writes to protected file (matched: {pattern.pattern})"

    # Everything else is allowed (tests, linting, general shell commands)
    return True, "allowed"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: guard_autonomous_bash.py <command>", file=sys.stderr)
        return 1

    command = sys.argv[1]
    allowed, reason = check_command(command)

    if not allowed:
        print(f"BLOCKED: {reason}", file=sys.stderr)
        print(f"  command: {command}", file=sys.stderr)
        print(
            "  hint: This operation is restricted during autonomous execution. "
            "Use the session-step protocol for state mutations.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
