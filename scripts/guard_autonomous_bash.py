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


def _sanitize_command(command: str) -> str:
    """Sanitize a command string before processing.

    Strips null bytes, ANSI escape sequences, and collapses whitespace.
    """
    # Strip null bytes
    command = command.replace("\x00", "")
    # Strip ANSI escape sequences
    command = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", command)
    # Collapse whitespace (but preserve newlines for splitting)
    command = re.sub(r"[^\S\n]+", " ", command)
    return command.strip()


def _split_commands(command: str) -> list[str]:
    """Split a command string on shell operators (&&, ||, ;, |, newlines).

    Respects quoted strings — does not split inside '...' or "...".
    Returns a list of individual command segments.
    """
    segments: list[str] = []
    current: list[str] = []
    i = 0
    n = len(command)

    while i < n:
        ch = command[i]

        # Handle single-quoted strings
        if ch == "'":
            current.append(ch)
            i += 1
            while i < n and command[i] != "'":
                current.append(command[i])
                i += 1
            if i < n:
                current.append(command[i])  # closing quote
                i += 1
            continue

        # Handle double-quoted strings
        if ch == '"':
            current.append(ch)
            i += 1
            while i < n and command[i] != '"':
                if command[i] == "\\" and i + 1 < n:
                    current.append(command[i])
                    current.append(command[i + 1])
                    i += 2
                    continue
                current.append(command[i])
                i += 1
            if i < n:
                current.append(command[i])  # closing quote
                i += 1
            continue

        # Check for && or ||
        if ch in ("&", "|") and i + 1 < n and command[i + 1] == ch:
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            i += 2
            continue

        # Check for ; or | (single pipe) or newline
        if ch in (";", "\n"):
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            i += 1
            continue

        if ch == "|":
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            i += 1
            continue

        current.append(ch)
        i += 1

    # Don't forget the last segment
    segment = "".join(current).strip()
    if segment:
        segments.append(segment)

    return segments


def _extract_git_subcommand(command: str) -> str | None:
    """Extract the git subcommand from a command string.

    Handles flags with values like -c key=val and -C /path by skipping
    them properly. Returns the base subcommand (e.g. "branch", "commit").
    """
    # Tokenize the command to properly skip flags with values
    tokens = command.split()
    if not tokens:
        return None

    # Find the 'git' token
    git_idx = None
    for idx, token in enumerate(tokens):
        if token == "git":
            git_idx = idx
            break

    if git_idx is None:
        return None

    # Walk tokens after 'git' to find the subcommand
    i = git_idx + 1
    while i < len(tokens):
        token = tokens[i]
        # Skip flags
        if token.startswith("-"):
            # Flags that take a separate value argument: -c, -C, --git-dir, etc.
            if token in ("-c", "-C", "--git-dir", "--work-tree", "--namespace"):
                i += 2  # skip flag and its value
                continue
            # Flags with embedded value (-c key=val already handled above)
            # Short flags without value (-v, -n, etc.)
            i += 1
            continue
        # First non-flag token is the subcommand
        return token
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
        # Explicit write flags only — --global and --system are scope modifiers, not writes
        "write_prefixes": ["--unset", "--replace-all", "--add"],
        "read_prefixes": ["--get", "--list", "-l", "--get-all", "--get-regexp"],
        "default": "read",  # bare 'git config' shows help; 'git config key' is a read
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

    # Special handling for 'config': detect key-value writes via positional args
    if subcommand == "config":
        return _classify_config(args, entry)

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


def _classify_config(args: str, entry: dict) -> tuple[str | None, str]:
    """Classify 'git config' as read or write based on arguments.

    Writes: --unset, --replace-all, --add, or positional 'key value' pair.
    Reads: --get, --list, -l, --get-all, --get-regexp, bare, or single key.
    Scope modifiers (--global, --system, --local) are ignored for classification.
    """
    # Check explicit write flags
    for prefix in entry["write_prefixes"]:
        if prefix in args.split():
            return f"config {prefix}", "write"

    # Check explicit read flags
    for prefix in entry["read_prefixes"]:
        if prefix in args.split():
            return f"config {prefix}", "read"

    # Strip scope modifiers and flags to find positional args
    tokens = args.split()
    positional = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ("--global", "--system", "--local", "--worktree", "--file", "-f"):
            # --file and -f take a value
            if token in ("--file", "-f"):
                i += 2
            else:
                i += 1
            continue
        if token.startswith("-"):
            i += 1
            continue
        positional.append(token)
        i += 1

    # Two or more positional args = 'key value' write
    if len(positional) >= 2:
        return "config key=value", "write"

    # Single positional arg or no args = read (show value or help)
    return None, entry["default"]


def _is_git_read_only(subcommand: str) -> bool:
    """Check if a git subcommand is read-only (exact match)."""
    return subcommand in _GIT_READ_SUBCOMMANDS


def _is_git_write(subcommand: str) -> bool:
    """Check if a git subcommand is a write operation."""
    return subcommand in _GIT_WRITE_SUBCOMMANDS


def _check_single_command(command: str) -> tuple[bool, str]:
    """Check whether a single bash command (no chaining) is allowed.

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

        # Unknown git subcommand — fail-closed (block unknown operations)
        return False, f"blocked: unknown git subcommand '{subcommand}' is not allowed during autonomous execution"

    # Check for shell write patterns targeting protected files
    for pattern in _SHELL_WRITE_PATTERNS:
        if pattern.search(command):
            return False, f"blocked: shell command writes to protected file (matched: {pattern.pattern})"

    # Everything else is allowed (tests, linting, general shell commands)
    return True, "allowed"


def check_command(command: str) -> tuple[bool, str]:
    """Check whether a bash command is allowed.

    Handles command chaining (&&, ||, ;, |, newlines) by checking
    each segment independently. If ANY segment is blocked, the
    entire command is blocked.

    Returns:
        (allowed, reason) — allowed=True means command is permitted.
    """
    # Sanitize first
    command = _sanitize_command(command)

    if os.environ.get("FOUNDRY_GUARD_DISABLED") == "1":
        return True, "guard disabled via FOUNDRY_GUARD_DISABLED=1"

    # Split on command chaining operators
    segments = _split_commands(command)

    if not segments:
        return True, "empty command allowed"

    # Check each segment — block if ANY segment is blocked
    last_reason = "allowed"
    for segment in segments:
        allowed, reason = _check_single_command(segment)
        if not allowed:
            return False, reason
        last_reason = reason

    # All segments passed — return the last segment's reason for specificity
    return True, last_reason


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
