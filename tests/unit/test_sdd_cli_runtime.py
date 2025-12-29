"""Unit tests for SDD CLI runtime components.

Tests cover:
- Configuration and context management
- JSON-only output helpers
- Resilience wrappers (timeout, retry, interrupt handling)
- Structured logging hooks
- Feature flag bootstrap
"""

import json
import sys
import time
from pathlib import Path

import pytest

from foundry_mcp.cli.config import CLIContext, create_context
from foundry_mcp.core.spec import find_specs_directory
from foundry_mcp.cli.flags import (
    CLIFlagRegistry,
    apply_cli_flag_overrides,
    flags_for_discovery,
    get_cli_flags,
)
from foundry_mcp.cli.logging import (
    CLILogContext,
    CLILogger,
    cli_command,
    generate_request_id,
    get_cli_logger,
    get_request_id,
    set_request_id,
)
from foundry_mcp.cli.output import emit, emit_error, emit_success
from foundry_mcp.cli.registry import get_context, set_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    MEDIUM_TIMEOUT,
    SLOW_TIMEOUT,
    TimeoutException,
    cli_retryable,
    handle_keyboard_interrupt,
    with_sync_timeout,
)
from foundry_mcp.core.feature_flags import FlagState


class TestCLIOutput:
    """Tests for JSON-only output helpers."""

    def test_emit_outputs_json(self, capsys):
        """emit() outputs valid JSON to stdout."""
        data = {"key": "value", "number": 42}
        emit(data)
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result == data

    def test_emit_handles_nested_data(self, capsys):
        """emit() handles nested structures."""
        data = {"nested": {"deep": {"value": [1, 2, 3]}}}
        emit(data)
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result == data

    def test_emit_success_wraps_data(self, capsys):
        """emit_success() wraps data in success envelope."""
        data = {"result": "test"}
        emit_success(data)
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["success"] is True
        assert result["data"] == data
        assert result["error"] is None

    def test_emit_success_includes_meta(self, capsys):
        """emit_success() includes optional meta."""
        data = {"result": "test"}
        meta = {"version": "1.0"}
        emit_success(data, meta=meta)
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["meta"]["version"] == meta["version"]
        assert isinstance(result["meta"].get("request_id"), str)
        assert result["meta"]["request_id"].startswith("cli_")

    def test_emit_error_outputs_to_stderr(self, capsys):
        """emit_error() outputs to stderr and exits."""
        with pytest.raises(SystemExit) as exc_info:
            emit_error("Something failed", code="TEST_ERROR")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        result = json.loads(captured.err)
        assert result["success"] is False
        assert result["data"]["error_code"] == "TEST_ERROR"
        assert result["error"] == "Something failed"

    def test_emit_error_includes_details(self, capsys):
        """emit_error() includes optional details."""
        with pytest.raises(SystemExit):
            emit_error("Failed", code="ERR", details={"file": "test.py"})

        captured = capsys.readouterr()
        result = json.loads(captured.err)
        assert result["data"]["details"]["file"] == "test.py"


class TestCLIContext:
    """Tests for CLI configuration and context management."""

    def test_create_context_returns_context(self):
        """create_context() returns CLIContext instance."""
        ctx = create_context()
        assert isinstance(ctx, CLIContext)

    def test_create_context_with_specs_dir(self, tmp_path):
        """create_context() accepts specs_dir override."""
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        ctx = create_context(specs_dir=str(specs_dir))
        assert ctx.specs_dir == specs_dir

    def test_cli_context_specs_dir_resolution(self, tmp_path):
        """CLIContext resolves specs_dir from override."""
        specs_dir = tmp_path / "my-specs"
        specs_dir.mkdir()
        ctx = CLIContext(specs_dir=str(specs_dir))
        assert ctx.specs_dir == specs_dir

    def test_cli_context_without_override_uses_config(self, tmp_path):
        """CLIContext falls back to ServerConfig when no override."""
        ctx = CLIContext()
        # Should not raise, even if no specs dir found
        _ = ctx.specs_dir

    def test_get_set_context(self):
        """get_context/set_context work correctly."""
        ctx = create_context()
        set_context(ctx)
        retrieved = get_context()
        assert retrieved is ctx


class TestFindSpecsDirectory:
    """Tests for specs directory auto-detection."""

    def _make_valid_specs_dir(self, specs_dir: Path) -> None:
        """Create a valid specs directory with required subdirs."""
        specs_dir.mkdir(exist_ok=True)
        (specs_dir / "pending").mkdir()
        (specs_dir / "active").mkdir()
        (specs_dir / "completed").mkdir()
        (specs_dir / "archived").mkdir()

    def test_find_specs_with_provided_path(self, tmp_path):
        """Finds specs/ when provided path is given."""
        specs_dir = tmp_path / "specs"
        self._make_valid_specs_dir(specs_dir)

        result = find_specs_directory(str(tmp_path))
        assert result == specs_dir

    def test_find_specs_with_direct_path(self, tmp_path):
        """Finds specs when provided_path points to specs dir."""
        specs_dir = tmp_path / "specs"
        self._make_valid_specs_dir(specs_dir)

        result = find_specs_directory(str(specs_dir))
        assert result == specs_dir

    def test_returns_none_when_not_found(self, tmp_path):
        """Returns None when no valid specs directory found."""
        # Create specs dir without required subdirs
        (tmp_path / "specs").mkdir()
        result = find_specs_directory(str(tmp_path))
        assert result is None


class TestCLIResilience:
    """Tests for timeout, retry, and interrupt handling."""

    def test_timeout_constants_exist(self):
        """Timeout constants are defined."""
        assert FAST_TIMEOUT > 0
        assert MEDIUM_TIMEOUT > FAST_TIMEOUT
        assert SLOW_TIMEOUT > MEDIUM_TIMEOUT

    def test_with_sync_timeout_passes_on_fast_function(self):
        """with_sync_timeout passes when function completes quickly."""

        @with_sync_timeout(seconds=5.0)
        def fast_func():
            return "done"

        result = fast_func()
        assert result == "done"

    @pytest.mark.skipif(
        sys.platform == "win32" or "xdist" in sys.modules,
        reason="SIGALRM not available on Windows or incompatible with xdist",
    )
    def test_with_sync_timeout_raises_on_slow_function(self):
        """with_sync_timeout raises TimeoutException on slow function.

        Note: This test is skipped when running with pytest-xdist because
        SIGALRM doesn't work correctly with worker processes.

        Uses 1 second timeout because signal.alarm requires int (0.5 becomes 0).
        """

        @with_sync_timeout(seconds=1, error_message="Too slow!")
        def slow_func():
            time.sleep(3.0)
            return "done"

        with pytest.raises(TimeoutException) as exc_info:
            slow_func()

        assert "Too slow!" in str(exc_info.value)

    def test_cli_retryable_succeeds_on_first_try(self):
        """cli_retryable succeeds when function works first time."""
        call_count = 0

        @cli_retryable(max_retries=3)
        def succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeeds()
        assert result == "success"
        assert call_count == 1

    def test_cli_retryable_retries_on_failure(self):
        """cli_retryable retries on specified exceptions."""
        call_count = 0

        @cli_retryable(max_retries=3, delay=0.01, exceptions=(ValueError,))
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert call_count == 3

    def test_cli_retryable_gives_up_after_max_retries(self):
        """cli_retryable gives up after max_retries."""

        @cli_retryable(max_retries=2, delay=0.01, exceptions=(ValueError,))
        def always_fails():
            raise ValueError("always")

        with pytest.raises(ValueError):
            always_fails()

    def test_handle_keyboard_interrupt_catches_ctrl_c(self):
        """handle_keyboard_interrupt catches KeyboardInterrupt."""
        cleanup_called = False

        def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        @handle_keyboard_interrupt(cleanup=cleanup)
        def interrupted():
            raise KeyboardInterrupt()

        with pytest.raises(SystemExit) as exc_info:
            interrupted()

        assert exc_info.value.code == 130
        assert cleanup_called


class TestCLILogging:
    """Tests for structured logging hooks."""

    def test_generate_request_id_format(self):
        """Request ID has expected format."""
        request_id = generate_request_id()
        assert request_id.startswith("cli_")
        assert len(request_id) == 16  # "cli_" + 12 hex chars

    def test_request_id_is_unique(self):
        """Each request ID is unique."""
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_get_set_request_id(self):
        """get/set request_id work correctly."""
        set_request_id("test_123")
        assert get_request_id() == "test_123"

    def test_cli_log_context_sets_request_id(self):
        """CLILogContext sets and restores request ID."""
        original = get_request_id()

        with CLILogContext() as ctx:
            assert ctx.request_id.startswith("cli_")
            assert get_request_id() == ctx.request_id

        # Should restore original (empty string by default)
        assert get_request_id() == original

    def test_cli_log_context_custom_id(self):
        """CLILogContext accepts custom request ID."""
        with CLILogContext(request_id="custom_123") as ctx:
            assert ctx.request_id == "custom_123"
            assert get_request_id() == "custom_123"

    def test_get_cli_logger_returns_logger(self):
        """get_cli_logger returns CLILogger instance."""
        logger = get_cli_logger()
        assert isinstance(logger, CLILogger)

    def test_cli_command_decorator_sets_context(self):
        """@cli_command decorator sets logging context."""
        captured_id = None

        @cli_command("test_cmd", emit_metrics=False)
        def test_cmd():
            nonlocal captured_id
            captured_id = get_request_id()
            return "result"

        result = test_cmd()
        assert result == "result"
        assert captured_id is not None
        assert captured_id.startswith("cli_")

    def test_cli_command_decorator_handles_exceptions(self):
        """@cli_command decorator handles exceptions gracefully."""

        @cli_command("failing_cmd", emit_metrics=False)
        def failing_cmd():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            failing_cmd()


class TestCLIFlags:
    """Tests for feature flag bootstrap."""

    def test_cli_flag_registry_creation(self):
        """CLIFlagRegistry can be created."""
        registry = CLIFlagRegistry()
        assert registry is not None

    def test_register_cli_flag(self):
        """Can register CLI-specific flags."""
        registry = CLIFlagRegistry()
        registry.register_cli_flag(
            name="test_flag_unique",
            description="Test flag",
            default_enabled=True,
            state=FlagState.STABLE,
        )
        assert registry.is_enabled("test_flag_unique")

    def test_cli_flag_disabled_by_default(self):
        """Flags default to disabled."""
        registry = CLIFlagRegistry()
        registry.register_cli_flag(
            name="test_disabled_flag",
            description="Disabled by default",
            default_enabled=False,
            state=FlagState.BETA,
        )
        assert not registry.is_enabled("test_disabled_flag")

    def test_apply_overrides(self):
        """apply_overrides enables/disables flags."""
        registry = CLIFlagRegistry()
        registry.register_cli_flag(
            name="override_test",
            description="Test override",
            default_enabled=False,
        )

        assert not registry.is_enabled("override_test")

        registry.apply_overrides({"override_test": True})
        assert registry.is_enabled("override_test")

        registry.clear_overrides()
        assert not registry.is_enabled("override_test")

    def test_get_discovery_manifest(self):
        """get_discovery_manifest returns flag info."""
        registry = CLIFlagRegistry()
        registry.register_cli_flag(
            name="discoverable_flag",
            description="For discovery",
            default_enabled=True,
            state=FlagState.STABLE,
        )

        manifest = registry.get_discovery_manifest()
        assert "discoverable_flag" in manifest
        assert manifest["discoverable_flag"]["enabled"] is True
        assert manifest["discoverable_flag"]["state"] == "stable"
        assert manifest["discoverable_flag"]["description"] == "For discovery"

    def test_list_flags(self):
        """list_flags returns all registered flag names."""
        registry = CLIFlagRegistry()
        registry.register_cli_flag(name="flag_a", description="A")
        registry.register_cli_flag(name="flag_b", description="B")

        flags = registry.list_flags()
        assert "flag_a" in flags
        assert "flag_b" in flags

    def test_get_cli_flags_singleton(self):
        """get_cli_flags returns the same instance."""
        reg1 = get_cli_flags()
        reg2 = get_cli_flags()
        assert reg1 is reg2

    def test_apply_cli_flag_overrides_function(self):
        """apply_cli_flag_overrides helper works."""
        registry = get_cli_flags()
        registry.register_cli_flag(
            name="cli_override_test",
            description="Test",
            default_enabled=False,
        )

        apply_cli_flag_overrides(enable=["cli_override_test"])
        assert registry.is_enabled("cli_override_test")

        registry.clear_overrides()

    def test_flags_for_discovery_function(self):
        """flags_for_discovery returns manifest."""
        manifest = flags_for_discovery()
        assert isinstance(manifest, dict)


class TestIntegration:
    """Integration tests for CLI runtime components."""

    def test_context_with_output(self, capsys, tmp_path):
        """Context and output work together."""
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        ctx = create_context(specs_dir=str(specs_dir))
        set_context(ctx)

        emit_success({"specs_dir": str(ctx.specs_dir)})

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["success"] is True
        assert str(specs_dir) in result["data"]["specs_dir"]

    def test_logging_with_command_decorator(self):
        """Logging context works with command decorator."""
        results = []

        @cli_command("integrated_cmd", emit_metrics=False)
        def integrated_cmd():
            results.append(
                {
                    "request_id": get_request_id(),
                    "logger": get_cli_logger(),
                }
            )
            return "done"

        integrated_cmd()

        assert len(results) == 1
        assert results[0]["request_id"].startswith("cli_")
        assert isinstance(results[0]["logger"], CLILogger)

    def test_flags_with_context(self, tmp_path):
        """Feature flags work with CLI context."""
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        ctx = create_context(specs_dir=str(specs_dir))
        set_context(ctx)

        registry = get_cli_flags()
        registry.register_cli_flag(
            name="context_integrated",
            description="Works with context",
            default_enabled=True,
        )

        manifest = flags_for_discovery()
        assert "context_integrated" in manifest
