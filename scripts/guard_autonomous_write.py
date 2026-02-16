#!/usr/bin/env python3
"""Guard script for Claude Code hooks: blocks Write/Edit to protected paths.

Usage (in Claude Code hook configuration):
    command: python scripts/guard_autonomous_write.py "$FILE_PATH"

Exit codes:
    0 — write allowed
    1 — write blocked (prints reason to stderr)

Protected paths:
    - specs/**/*.json          (spec tampering prevention)
    - **/foundry-mcp.toml      (config self-escalation prevention)
    - **/.foundry-mcp.toml     (config self-escalation prevention)
    - .foundry-mcp/sessions/   (session state tampering prevention)
    - .foundry-mcp/journals/   (audit trail tampering prevention)
    - .foundry-mcp/proofs/     (proof record tampering prevention)
    - .foundry-mcp/audit/      (audit ledger tampering prevention)

Environment variables:
    FOUNDRY_GUARD_DISABLED=1   — bypass all checks (emergency escape hatch)
    FOUNDRY_GUARD_EXTRA_BLOCKED — colon-separated additional path prefixes to block
    FOUNDRY_GUARD_EXTRA_ALLOWED — colon-separated additional path prefixes to allow
                                  (evaluated before block rules)
"""
import os
import sys
from pathlib import PurePosixPath
from typing import Callable


def _normalize(path: str) -> str:
    """Normalize path for comparison: resolve, lowercase on case-insensitive fs."""
    try:
        resolved = os.path.realpath(path)
    except (OSError, ValueError):
        resolved = path
    return resolved


# Patterns that block writes.  Each entry is (description, match_function).
_BLOCKED_PATTERNS: list[tuple[str, Callable[[str], bool]]] = []


def _matches_spec_json(path: str) -> bool:
    """Block writes to spec JSON files under any specs/ directory."""
    parts = PurePosixPath(path).parts
    for i, part in enumerate(parts):
        if part == "specs" and i < len(parts) - 1:
            if parts[-1].endswith(".json"):
                return True
    return False


def _matches_config_toml(path: str) -> bool:
    """Block writes to foundry-mcp config files."""
    basename = os.path.basename(path)
    return basename in ("foundry-mcp.toml", ".foundry-mcp.toml")


def _matches_session_state(path: str) -> bool:
    """Block writes to session state directory."""
    norm = path.replace("\\", "/")
    return "/.foundry-mcp/sessions/" in norm or norm.startswith(".foundry-mcp/sessions/")


def _matches_journal_dir(path: str) -> bool:
    """Block writes to journal/audit directories."""
    norm = path.replace("\\", "/")
    protected_dirs = [
        "/.foundry-mcp/journals/",
        "/.foundry-mcp/audit/",
        "/.foundry-mcp/proofs/",
    ]
    for d in protected_dirs:
        if d in norm or norm.startswith(d.lstrip("/")):
            return True
    return False


_BLOCKED_PATTERNS = [
    ("spec file (specs/**/*.json)", _matches_spec_json),
    ("config file (foundry-mcp.toml)", _matches_config_toml),
    ("session state (.foundry-mcp/sessions/)", _matches_session_state),
    ("journal/audit directory (.foundry-mcp/journals|audit|proofs/)", _matches_journal_dir),
]


def check_path(file_path: str) -> tuple[bool, str]:
    """Check whether a write to file_path is allowed.

    Returns:
        (allowed, reason) — allowed=True means write is permitted.
    """
    if os.environ.get("FOUNDRY_GUARD_DISABLED") == "1":
        return True, "guard disabled via FOUNDRY_GUARD_DISABLED=1"

    normalized = _normalize(file_path)

    # Check extra allowed paths (takes precedence over block rules)
    extra_allowed = os.environ.get("FOUNDRY_GUARD_EXTRA_ALLOWED", "")
    if extra_allowed:
        for prefix in extra_allowed.split(":"):
            prefix = prefix.strip()
            if prefix and normalized.startswith(_normalize(prefix)):
                return True, f"explicitly allowed via FOUNDRY_GUARD_EXTRA_ALLOWED ({prefix})"

    # Check extra blocked paths
    extra_blocked = os.environ.get("FOUNDRY_GUARD_EXTRA_BLOCKED", "")
    if extra_blocked:
        for prefix in extra_blocked.split(":"):
            prefix = prefix.strip()
            if prefix and normalized.startswith(_normalize(prefix)):
                return False, f"blocked via FOUNDRY_GUARD_EXTRA_BLOCKED ({prefix})"

    # Check built-in block patterns
    for description, matcher in _BLOCKED_PATTERNS:
        if matcher(file_path) or matcher(normalized):
            return False, f"blocked: write targets protected {description}"

    return True, "allowed"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: guard_autonomous_write.py <file_path>", file=sys.stderr)
        return 1

    file_path = sys.argv[1]
    allowed, reason = check_path(file_path)

    if not allowed:
        print(f"BLOCKED: {reason}", file=sys.stderr)
        print(f"  path: {file_path}", file=sys.stderr)
        print(
            "  hint: This file is protected during autonomous execution. "
            "Use MCP protocol actions to modify this resource.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
