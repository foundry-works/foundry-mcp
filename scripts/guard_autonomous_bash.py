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


# Git subcommands that are always write operations (blocked by default).
# Subcommands whose read/write classification depends on arguments
# (branch, remote, stash, config) are handled by _COMPOUND_* tables instead.
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
    "tag",
}

# Git subcommands that are always read-only (always allowed).
# For subcommands with mixed read/write behaviour, see _COMPOUND_* tables.
_GIT_READ_SUBCOMMANDS = {
    "status",
    "diff",
    "log",
    "show",
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
    """Extract the git subcommand from a command string.

    Returns the base subcommand (e.g. "branch", "remote", "commit").
    """
    match = re.search(r"\bgit\s+(?:-\w+\s+)*(\S+)", command)
    if match:
        return match.group(1).strip()
    return None


def _extract_git_args_after_subcommand(command: str, subcommand: str) -> str:
    """Extract the arguments following 'git <subcommand>' for compound checking.

    Returns the remainder of the command after the subcommand, stripped.
    """
    pattern = re.compile(rf"\bgit\s+(?:-\w+\s+)*{re.escape(subcommand)}\b\s*(.*)", re.DOTALL)
    match = pattern.search(command)
    if match:
        return match.group(1).strip()
    return ""


# Compound subcommand classification tables.
# Subcommands listed here have mixed read/write behaviour depending on
# the arguments that follow. When the base subcommand is in this table
# it is classified exclusively via these patterns — the simple read/write
# sets above are skipped.
#
# "write_prefixes": argument prefixes that make the operation a write.
# "read_prefixes": argument prefixes that are explicitly read-only.
# "default": disposition when no prefix matches ("read" or "write").
_COMPOUND_SUBCOMMANDS: dict[str, dict] = {
    "branch": {
        "write_prefixes": ["-d", "-D", "-m", "-M", "--delete", "--move"],
        "read_prefixes": ["-a", "-r", "-v", "--list", "--contains", "--merged", "--no-merged"],
        "default": "read",  # bare 'git branch' lists branches
    },
    "remote": {
        "write_prefixes": ["add", "remove", "set-url", "rename", "rm", "set-head", "prune"],
        "read_prefixes": ["-v", "show", "get-url"],
        "default": "read",  # bare 'git remote' lists remotes
    },
    "stash": {
        "write_prefixes": ["drop", "pop", "clear", "apply", "push", "save"],
        "read_prefixes": ["list", "show"],
        "default": "write",  # bare 'git stash' = stash push (write)
    },
    "config": {
        "write_prefixes": ["--global", "--system", "--unset", "--replace-all", "--add"],
        "read_prefixes": ["--get", "--list", "-l", "--get-all", "--get-regexp"],
        "default": "read",  # bare 'git config' shows help
    },
}


def _classify_compound(subcommand: str, args: str) -> tuple[str | None, str]:
    """Classify a compound subcommand as read or write.

    Returns (matched_pattern | None, disposition) where disposition is
    "read", "write", or "unknown" (not a compound subcommand).
    """
    entry = _COMPOUND_SUBCOMMANDS.get(subcommand)
    if entry is None:
        return None, "unknown"

    # Check writes first (more specific)
    for prefix in entry["write_prefixes"]:
        if args.startswith(prefix):
            return f"{subcommand} {prefix}", "write"

    # Check reads
    for prefix in entry["read_prefixes"]:
        if args.startswith(prefix):
            return f"{subcommand} {prefix}", "read"

    # No args or unrecognised args → use default
    return None, entry["default"]


def _is_git_read_only(subcommand: str) -> bool:
    """Check if a git subcommand is read-only."""
    for read_cmd in _GIT_READ_SUBCOMMANDS:
        if subcommand.startswith(read_cmd):
            return True
    return False


def _is_git_write(subcommand: str) -> bool:
    """Check if a git subcommand is a write operation."""
    for write_cmd in _GIT_WRITE_SUBCOMMANDS:
        if subcommand == write_cmd:
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

        # Check compound subcommands FIRST (branch, remote, stash, config).
        # These have both read and write variants based on arguments.
        args = _extract_git_args_after_subcommand(command, subcommand)
        matched_pattern, disposition = _classify_compound(subcommand, args)
        if disposition == "write":
            label = matched_pattern or subcommand
            return False, f"blocked: git write operation '{label}' is not allowed during autonomous execution"
        elif disposition == "read":
            label = matched_pattern or subcommand
            return True, f"git read-only operation: {label}"
        # disposition == "unknown" → not a compound subcommand, fall through

        # Check simple read-only subcommands
        if _is_git_read_only(subcommand):
            return True, f"git read-only operation: {subcommand}"

        # Check simple write operations
        if _is_git_write(subcommand):
            # Special case: allow git commit when explicitly permitted
            if subcommand == "commit" and os.environ.get("FOUNDRY_GUARD_ALLOW_GIT_COMMIT") == "1":
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
