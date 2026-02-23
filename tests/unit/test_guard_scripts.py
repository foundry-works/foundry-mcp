"""Tests for guard scripts: guard_autonomous_bash.py and guard_autonomous_write.py.

Coverage:
- Bash guard: git read/write classification, env var bypasses, edge cases
- Write guard: protected path blocking, normal path allowing, env var precedence
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Import the check functions directly from the guard scripts
from scripts.guard_autonomous_bash import check_command
from scripts.guard_autonomous_write import check_path


# =============================================================================
# Bash Guard: Read-Only Git Commands
# =============================================================================


class TestBashGuardReadOnly:
    """Git read-only commands should always be allowed."""

    @pytest.mark.parametrize(
        "command",
        [
            "git status",
            "git diff",
            "git diff --cached",
            "git log --oneline -10",
            "git show HEAD",
            "git branch",
            "git branch -a",
            "git remote -v",
            "git describe --tags",
            "git rev-parse HEAD",
            "git ls-files",
            "git ls-tree HEAD",
            "git cat-file -p HEAD",
            "git reflog",
            "git shortlog -sn",
            "git blame src/main.py",
            "git grep 'pattern'",
            "git stash list",
            "git stash show",
            "git config --get user.name",
            "git config --list",
            "git config -l",
        ],
    )
    def test_git_read_commands_allowed(self, command):
        allowed, reason = check_command(command)
        assert allowed is True, f"Expected allowed for '{command}', got blocked: {reason}"

    def test_git_diff_with_flags(self):
        allowed, _ = check_command("git diff --stat HEAD~3..HEAD")
        assert allowed is True

    def test_git_log_with_format(self):
        allowed, _ = check_command("git log --format='%H %s' -5")
        assert allowed is True


# =============================================================================
# Bash Guard: Write Commands Blocked
# =============================================================================


class TestBashGuardWriteBlocked:
    """Git write commands should be blocked by default."""

    @pytest.mark.parametrize(
        "command",
        [
            "git commit -m 'test'",
            "git push origin main",
            "git push",
            "git reset --hard HEAD~1",
            "git reset HEAD~1",
            "git rebase main",
            "git rebase -i HEAD~3",
            "git checkout main",
            "git clean -fd",
            "git merge feature-branch",
            "git cherry-pick abc123",
            "git revert HEAD",
            "git stash drop",
            "git tag v1.0",
        ],
    )
    def test_git_write_commands_blocked(self, command):
        allowed, reason = check_command(command)
        assert allowed is False, f"Expected blocked for '{command}', got allowed: {reason}"
        assert "blocked" in reason.lower()

    @pytest.mark.parametrize(
        "command",
        [
            "git branch -d old-branch",
            "git branch -D old-branch",
            "git branch -m old new",
            "git branch -M old new",
            "git remote add upstream url",
            "git remote remove origin",
            "git remote set-url origin new-url",
        ],
    )
    def test_compound_write_subcommands_blocked(self, command):
        """Compound subcommands (branch -D, remote add) are correctly blocked
        as write operations despite the base command being in the read list."""
        allowed, reason = check_command(command)
        assert allowed is False, f"Expected blocked for '{command}', got allowed: {reason}"
        assert "blocked" in reason.lower()


# =============================================================================
# Bash Guard: Environment Variable Bypasses
# =============================================================================


class TestBashGuardBypass:
    """Environment variable overrides for bash guard."""

    def test_guard_disabled_allows_everything(self):
        with patch.dict(os.environ, {"FOUNDRY_GUARD_DISABLED": "1"}):
            allowed, reason = check_command("git push --force origin main")
            assert allowed is True
            assert "disabled" in reason.lower()

    def test_guard_disabled_zero_does_not_bypass(self):
        with patch.dict(os.environ, {"FOUNDRY_GUARD_DISABLED": "0"}, clear=False):
            allowed, _ = check_command("git push origin main")
            assert allowed is False

    def test_allow_git_commit_env(self):
        with patch.dict(os.environ, {"FOUNDRY_GUARD_ALLOW_GIT_COMMIT": "1"}):
            allowed, reason = check_command("git commit -m 'allowed'")
            assert allowed is True
            assert "FOUNDRY_GUARD_ALLOW_GIT_COMMIT" in reason

    def test_allow_git_commit_does_not_allow_push(self):
        with patch.dict(os.environ, {"FOUNDRY_GUARD_ALLOW_GIT_COMMIT": "1"}):
            allowed, _ = check_command("git push origin main")
            assert allowed is False


# =============================================================================
# Bash Guard: Edge Cases
# =============================================================================


class TestBashGuardEdgeCases:
    """Edge cases: unknown commands, flags before subcommand, shell patterns."""

    def test_bare_git_allowed(self):
        """Bare 'git' or 'git --version' should be allowed."""
        allowed, _ = check_command("git --version")
        assert allowed is True

    def test_unknown_git_subcommand_allowed(self):
        """Unknown git subcommands are allowed (fail-open for reads)."""
        allowed, _ = check_command("git some-unknown-command")
        assert allowed is True

    def test_non_git_commands_allowed(self):
        """Non-git commands should be allowed."""
        allowed, _ = check_command("pytest tests/ -v")
        assert allowed is True

        allowed, _ = check_command("python -m pytest")
        assert allowed is True

        allowed, _ = check_command("ls -la")
        assert allowed is True

    def test_shell_write_to_spec_blocked(self):
        """Shell writes to spec files should be blocked."""
        allowed, _ = check_command("echo 'data' > specs/test.json")
        assert allowed is False

    def test_shell_write_to_config_blocked(self):
        """Shell writes to config TOML should be blocked."""
        allowed, _ = check_command("echo 'data' >> foundry-mcp.toml")
        assert allowed is False

    def test_rm_audit_files_blocked(self):
        """Removing audit files should be blocked."""
        allowed, _ = check_command("rm .foundry-mcp/journals/entry.json")
        assert allowed is False

    def test_sed_inplace_spec_blocked(self):
        """In-place sed edits of spec files should be blocked."""
        allowed, _ = check_command("sed -i 's/old/new/' specs/test.json")
        assert allowed is False

    def test_stash_list_and_show_still_read_only(self):
        """git stash list/show remain allowed as read-only."""
        allowed, _ = check_command("git stash list")
        assert allowed is True
        allowed, _ = check_command("git stash show")
        assert allowed is True

    def test_stash_drop_and_pop_blocked(self):
        """Stash write operations should be blocked."""
        allowed, _ = check_command("git stash drop")
        assert allowed is False
        allowed, _ = check_command("git stash pop")
        assert allowed is False
        allowed, _ = check_command("git stash clear")
        assert allowed is False

    def test_bare_branch_and_remote_still_allowed(self):
        """Plain 'git branch' and 'git remote' (no args) remain read-only."""
        allowed, _ = check_command("git branch")
        assert allowed is True
        allowed, _ = check_command("git branch -a")
        assert allowed is True
        allowed, _ = check_command("git remote -v")
        assert allowed is True

    def test_git_with_flags_before_subcommand(self):
        """Git commands with flags before the subcommand.

        The regex ``(?:-\\w+\\s+)*`` only matches bare short flags (e.g. -v, -n).
        Flags with values (e.g. -C /path, -c key=val) cause the regex to
        capture the value argument as the subcommand instead.
        """
        # Bare short flags are handled correctly
        allowed, _ = check_command("git commit -m 'msg'")
        assert allowed is False

        # Flags with values (-c key=val) cause subcommand misdetection:
        # the value "core.pager=cat" is extracted as the subcommand
        allowed, _ = check_command("git -c core.pager=cat commit -m 'msg'")
        # This is allowed because "core.pager=cat commit" is unknown → allowed
        assert allowed is True

    def test_cp_to_protected_path_blocked(self):
        """Copying files to protected paths should be blocked."""
        allowed, _ = check_command("cp bad.json specs/real.json")
        assert allowed is False


# =============================================================================
# Write Guard: Protected Paths Blocked
# =============================================================================


class TestWriteGuardBlocked:
    """Write guard should block writes to protected paths."""

    @pytest.mark.parametrize(
        "path",
        [
            "specs/active/my-spec.json",
            "/workspace/specs/active/my-spec.json",
            "specs/completed/old.json",
            "foundry-mcp.toml",
            ".foundry-mcp.toml",
            "/workspace/foundry-mcp.toml",
            ".foundry-mcp/sessions/sess-001.json",
            "/workspace/.foundry-mcp/sessions/sess-001.json",
            ".foundry-mcp/journals/entry.json",
            ".foundry-mcp/audit/event.json",
            ".foundry-mcp/proofs/proof-001.json",
        ],
    )
    def test_protected_paths_blocked(self, path):
        allowed, reason = check_path(path)
        assert allowed is False, f"Expected blocked for '{path}', got allowed: {reason}"
        assert "blocked" in reason.lower()


# =============================================================================
# Write Guard: Normal Paths Allowed
# =============================================================================


class TestWriteGuardAllowed:
    """Write guard should allow writes to normal source/test files."""

    @pytest.mark.parametrize(
        "path",
        [
            "src/main.py",
            "src/foundry_mcp/core/utils.py",
            "tests/unit/test_something.py",
            "/tmp/scratch.txt",
            "README.md",
            "pyproject.toml",
            "src/config.py",
        ],
    )
    def test_normal_paths_allowed(self, path):
        allowed, reason = check_path(path)
        assert allowed is True, f"Expected allowed for '{path}', got blocked: {reason}"


# =============================================================================
# Write Guard: Environment Variable Overrides
# =============================================================================


class TestWriteGuardBypass:
    """Environment variable overrides for write guard."""

    def test_guard_disabled_allows_everything(self):
        with patch.dict(os.environ, {"FOUNDRY_GUARD_DISABLED": "1"}):
            allowed, reason = check_path("specs/active/my-spec.json")
            assert allowed is True
            assert "disabled" in reason.lower()

    def test_extra_blocked_blocks_custom_path(self):
        with patch.dict(os.environ, {"FOUNDRY_GUARD_EXTRA_BLOCKED": "/custom/protected"}):
            allowed, reason = check_path("/custom/protected/file.txt")
            assert allowed is False
            assert "EXTRA_BLOCKED" in reason

    def test_extra_allowed_overrides_block(self):
        """EXTRA_ALLOWED takes precedence over built-in block rules."""
        with patch.dict(os.environ, {"FOUNDRY_GUARD_EXTRA_ALLOWED": "specs/active/my-spec.json"}):
            allowed, reason = check_path("specs/active/my-spec.json")
            assert allowed is True
            assert "EXTRA_ALLOWED" in reason

    def test_extra_allowed_precedence_over_extra_blocked(self):
        """EXTRA_ALLOWED is evaluated before EXTRA_BLOCKED."""
        with patch.dict(
            os.environ,
            {
                "FOUNDRY_GUARD_EXTRA_ALLOWED": "/shared/path",
                "FOUNDRY_GUARD_EXTRA_BLOCKED": "/shared/path",
            },
        ):
            allowed, reason = check_path("/shared/path/file.txt")
            assert allowed is True
            assert "EXTRA_ALLOWED" in reason
