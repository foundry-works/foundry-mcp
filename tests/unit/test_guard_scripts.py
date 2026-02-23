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

    def test_unknown_git_subcommand_blocked(self):
        """Unknown git subcommands are blocked (fail-closed)."""
        allowed, _ = check_command("git some-unknown-command")
        assert allowed is False

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

        The token-based parser correctly skips -c key=val flags
        and finds the real subcommand.
        """
        # Bare short flags are handled correctly
        allowed, _ = check_command("git commit -m 'msg'")
        assert allowed is False

        # Flags with values (-c key=val) are correctly skipped;
        # the real subcommand "commit" is detected and blocked
        allowed, _ = check_command("git -c core.pager=cat commit -m 'msg'")
        assert allowed is False

    def test_cp_to_protected_path_blocked(self):
        """Copying files to protected paths should be blocked."""
        allowed, _ = check_command("cp bad.json specs/real.json")
        assert allowed is False

    def test_git_fetch_allowed(self):
        """git fetch is read-only from working tree perspective."""
        allowed, _ = check_command("git fetch origin")
        assert allowed is True
        allowed, _ = check_command("git fetch --all")
        assert allowed is True

    def test_git_pull_blocked(self):
        """git pull modifies working tree and is blocked."""
        allowed, _ = check_command("git pull origin main")
        assert allowed is False

    def test_git_add_blocked(self):
        """git add stages files and is blocked."""
        allowed, _ = check_command("git add .")
        assert allowed is False
        allowed, _ = check_command("git add -A")
        assert allowed is False

    def test_git_rm_mv_blocked(self):
        """git rm and git mv are write operations."""
        allowed, _ = check_command("git rm src/old.py")
        assert allowed is False
        allowed, _ = check_command("git mv src/old.py src/new.py")
        assert allowed is False

    def test_git_restore_switch_blocked(self):
        """git restore and git switch modify working tree."""
        allowed, _ = check_command("git restore --staged .")
        assert allowed is False
        allowed, _ = check_command("git switch main")
        assert allowed is False

    def test_empty_command(self):
        """Empty and whitespace-only commands are allowed."""
        allowed, _ = check_command("")
        assert allowed is True
        allowed, _ = check_command("   ")
        assert allowed is True

    def test_multiple_spaces_between_git_args(self):
        """Multiple spaces between git args should not affect detection."""
        allowed, _ = check_command("git    push   origin   main")
        assert allowed is False


# =============================================================================
# Bash Guard: Shell Wrapper Bypass Prevention
# =============================================================================


class TestBashGuardShellWrappers:
    """Shell wrapper commands that could bypass the git guard must be blocked."""

    @pytest.mark.parametrize(
        "command",
        [
            'bash -c "git push origin main"',
            "sh -c 'git push origin main'",
            "zsh -c 'git reset --hard HEAD'",
            "eval 'git push'",
            "eval git push",
            "exec git push",
        ],
    )
    def test_shell_wrapper_git_blocked(self, command):
        """Shell wrappers executing git write commands are blocked."""
        allowed, reason = check_command(command)
        assert allowed is False, f"Expected blocked for '{command}', got allowed: {reason}"
        assert "blocked" in reason.lower()

    def test_bash_without_c_flag_allowed(self):
        """Plain bash without -c flag is not a wrapper pattern."""
        allowed, _ = check_command("bash script.sh")
        assert allowed is True

    def test_python_c_with_git_blocked(self):
        """python -c with git reference is blocked."""
        allowed, _ = check_command('python3 -c "import subprocess; subprocess.run([\'git\', \'push\'])"')
        assert allowed is False


# =============================================================================
# Bash Guard: Full-Path Git Binary
# =============================================================================


class TestBashGuardFullPathGit:
    """Full-path git binaries like /usr/bin/git must be detected."""

    def test_full_path_git_push_blocked(self):
        allowed, _ = check_command("/usr/bin/git push origin main")
        assert allowed is False

    def test_full_path_git_status_allowed(self):
        allowed, _ = check_command("/usr/bin/git status")
        assert allowed is True

    def test_full_path_git_commit_blocked(self):
        allowed, _ = check_command("/usr/local/bin/git commit -m 'msg'")
        assert allowed is False


# =============================================================================
# Bash Guard: Compound Subcommand Arg Extraction
# =============================================================================


class TestBashGuardCompoundArgExtraction:
    """Git flags before subcommands (e.g. -C /path) must not break classification."""

    def test_git_C_flag_branch_delete_blocked(self):
        """git -C /tmp branch -D main should be blocked."""
        allowed, _ = check_command("git -C /tmp branch -D main")
        assert allowed is False

    def test_git_c_config_before_push_blocked(self):
        """git -c core.pager=cat push should be blocked."""
        allowed, _ = check_command("git -c core.pager=cat push origin main")
        assert allowed is False

    def test_git_git_dir_flag_before_status_allowed(self):
        """git --git-dir /foo status should be allowed."""
        allowed, _ = check_command("git --git-dir /foo status")
        assert allowed is True


# =============================================================================
# Bash Guard: Bisect Classification
# =============================================================================


class TestBashGuardBisect:
    """git bisect subcommand classification (compound)."""

    @pytest.mark.parametrize(
        "command",
        [
            "git bisect start",
            "git bisect good",
            "git bisect bad",
            "git bisect reset",
            "git bisect skip",
            "git bisect run make test",
        ],
    )
    def test_bisect_write_operations_blocked(self, command):
        allowed, reason = check_command(command)
        assert allowed is False, f"Expected blocked for '{command}', got allowed: {reason}"

    @pytest.mark.parametrize(
        "command",
        [
            "git bisect log",
            "git bisect visualize",
        ],
    )
    def test_bisect_read_operations_allowed(self, command):
        allowed, reason = check_command(command)
        assert allowed is True, f"Expected allowed for '{command}', got blocked: {reason}"


# =============================================================================
# Bash Guard: Extended Sanitization
# =============================================================================


class TestBashGuardExtendedSanitization:
    """Extended sanitization: OSC, DCS, C1 codes."""

    def test_osc_hyperlink_stripped(self):
        """OSC hyperlink sequences should be stripped."""
        cmd = "\x1b]8;;http://example.com\x1b\\git status\x1b]8;;\x1b\\"
        allowed, _ = check_command(cmd)
        assert allowed is True

    def test_c1_control_codes_stripped(self):
        """8-bit C1 control codes should be stripped."""
        cmd = "\x9bgit status\x9b0m"
        allowed, _ = check_command(cmd)
        assert allowed is True


# =============================================================================
# Bash Guard: Command Chaining
# =============================================================================


class TestBashGuardCommandChaining:
    """Command chaining should check each segment independently."""

    def test_safe_and_blocked_via_double_ampersand(self):
        """git status && git push → blocked (push is write)."""
        allowed, reason = check_command("git status && git push")
        assert allowed is False
        assert "blocked" in reason.lower()

    def test_safe_and_safe_allowed(self):
        """git status && git log → allowed."""
        allowed, _ = check_command("git status && git log --oneline")
        assert allowed is True

    def test_blocked_via_or(self):
        """git status || git push → blocked."""
        allowed, _ = check_command("git status || git push")
        assert allowed is False

    def test_blocked_via_semicolon(self):
        """git log; git push → blocked."""
        allowed, _ = check_command("git log; git push origin main")
        assert allowed is False

    def test_pipe_read_only_allowed(self):
        """git log | head → allowed (both are read-only)."""
        allowed, _ = check_command("git log | head -20")
        assert allowed is True

    def test_newline_splitting(self):
        """Commands separated by newlines are checked independently."""
        allowed, _ = check_command("git status\ngit push")
        assert allowed is False

    def test_quoted_semicolons_not_split(self):
        """Semicolons inside quotes should NOT cause splitting."""
        allowed, _ = check_command("git log --format='%H;%s'")
        assert allowed is True


# =============================================================================
# Bash Guard: Sanitization
# =============================================================================


class TestBashGuardSanitization:
    """Input sanitization: null bytes, ANSI escapes."""

    def test_null_bytes_stripped(self):
        """Null bytes in command are stripped before processing."""
        allowed, _ = check_command("git\x00 status")
        assert allowed is True

    def test_ansi_escapes_stripped(self):
        """ANSI escape sequences are stripped."""
        allowed, _ = check_command("\x1b[31mgit status\x1b[0m")
        assert allowed is True


# =============================================================================
# Bash Guard: Git Config Classification
# =============================================================================


class TestBashGuardGitConfig:
    """Git config read/write classification."""

    def test_config_get_allowed(self):
        allowed, _ = check_command("git config --get user.name")
        assert allowed is True

    def test_config_list_allowed(self):
        allowed, _ = check_command("git config --list")
        assert allowed is True

    def test_config_global_read_allowed(self):
        """git config --global user.name (single key = read) → allowed."""
        allowed, _ = check_command("git config --global user.name")
        assert allowed is True

    def test_config_global_write_blocked(self):
        """git config user.name 'Tyler' (key + value = write) → blocked."""
        allowed, reason = check_command('git config user.name "Tyler"')
        assert allowed is False
        assert "blocked" in reason.lower()

    def test_config_global_key_value_blocked(self):
        """git config --global user.name 'Tyler' → blocked (write)."""
        allowed, _ = check_command("git config --global user.name Tyler")
        assert allowed is False

    def test_config_unset_blocked(self):
        """git config --unset key → blocked."""
        allowed, _ = check_command("git config --unset user.name")
        assert allowed is False


# =============================================================================
# Bash Guard: Fail-Closed for Unknown Subcommands
# =============================================================================


class TestBashGuardFailClosed:
    """Unknown git subcommands are blocked (fail-closed)."""

    def test_filter_branch_blocked(self):
        allowed, _ = check_command("git filter-branch")
        assert allowed is False

    def test_update_ref_blocked(self):
        allowed, _ = check_command("git update-ref")
        assert allowed is False

    def test_replace_blocked(self):
        allowed, _ = check_command("git replace")
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
