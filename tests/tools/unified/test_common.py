"""Unit tests for the shared tool helpers in common.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from foundry_mcp.core.responses.types import ErrorCode
from foundry_mcp.tools.unified.common import (
    build_request_id,
    dispatch_with_standard_errors,
    make_metric_name,
    make_validation_error_fn,
    resolve_specs_dir,
)
from foundry_mcp.tools.unified.router import ActionRouter


# Helper to create mock authorization results
@dataclass
class MockAuthzResult:
    """Mock authorization result for testing."""

    allowed: bool
    denied_action: str = ""
    configured_role: str = "observer"
    required_role: str = ""


# -----------------------------------------------------------------------
# 1. build_request_id
# -----------------------------------------------------------------------


class TestBuildRequestId:
    """Tests for build_request_id()."""

    def test_returns_existing_correlation_id(self):
        with patch(
            "foundry_mcp.tools.unified.common.get_correlation_id",
            return_value="existing_abc123",
        ):
            assert build_request_id("task") == "existing_abc123"

    def test_generates_new_id_with_prefix(self):
        with (
            patch(
                "foundry_mcp.tools.unified.common.get_correlation_id",
                return_value=None,
            ),
            patch(
                "foundry_mcp.tools.unified.common.generate_correlation_id",
                return_value="task_deadbeef1234",
            ) as mock_gen,
        ):
            result = build_request_id("task")
            assert result == "task_deadbeef1234"
            mock_gen.assert_called_once_with(prefix="task")

    def test_different_prefixes(self):
        with (
            patch(
                "foundry_mcp.tools.unified.common.get_correlation_id",
                return_value=None,
            ),
            patch(
                "foundry_mcp.tools.unified.common.generate_correlation_id",
                side_effect=lambda prefix: f"{prefix}_abc",
            ),
        ):
            assert build_request_id("authoring") == "authoring_abc"
            assert build_request_id("server") == "server_abc"


# -----------------------------------------------------------------------
# 2. make_metric_name
# -----------------------------------------------------------------------


class TestMakeMetricName:
    """Tests for make_metric_name()."""

    def test_simple_action(self):
        assert make_metric_name("authoring", "create") == "authoring.create"

    def test_hyphenated_action(self):
        assert make_metric_name("authoring", "phase-add") == "authoring.phase_add"

    def test_already_underscored(self):
        assert make_metric_name("task", "phase_add") == "task.phase_add"

    def test_compound_prefix(self):
        assert make_metric_name("unified_tools.task", "prepare") == "unified_tools.task.prepare"

    def test_multiple_hyphens(self):
        assert make_metric_name("test", "run-all-suites") == "test.run_all_suites"


# -----------------------------------------------------------------------
# 3. resolve_specs_dir
# -----------------------------------------------------------------------


class TestResolveSpecsDir:
    """Tests for resolve_specs_dir()."""

    def test_with_explicit_path(self, tmp_path):
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        config = MagicMock(specs_dir=None)

        with patch(
            "foundry_mcp.tools.unified.common.find_specs_directory",
            return_value=specs_dir,
        ):
            result_dir, error = resolve_specs_dir(config, str(specs_dir))
            assert result_dir == specs_dir
            assert error is None

    def test_with_config_path_object(self):
        config = MagicMock(specs_dir=Path("/some/specs"))
        result_dir, error = resolve_specs_dir(config)
        assert result_dir == Path("/some/specs")
        assert error is None

    def test_with_config_string(self):
        config = MagicMock(specs_dir="/some/specs")
        result_dir, error = resolve_specs_dir(config)
        assert result_dir == Path("/some/specs")
        assert error is None

    def test_falls_back_to_auto_detect(self):
        config = MagicMock(specs_dir=None)
        with patch(
            "foundry_mcp.tools.unified.common.find_specs_directory",
            return_value=Path("/auto/detected"),
        ):
            result_dir, error = resolve_specs_dir(config)
            assert result_dir == Path("/auto/detected")
            assert error is None

    def test_exception_returns_error_dict(self):
        config = MagicMock(specs_dir=None)
        with patch(
            "foundry_mcp.tools.unified.common.find_specs_directory",
            side_effect=RuntimeError("boom"),
        ):
            result_dir, error = resolve_specs_dir(config)
            assert result_dir is None
            assert error is not None
            assert error["success"] is False
            assert "boom" in error["error"]

    def test_none_specs_dir_and_no_auto_detect(self):
        config = MagicMock(specs_dir=None)
        with patch(
            "foundry_mcp.tools.unified.common.find_specs_directory",
            return_value=None,
        ):
            result_dir, error = resolve_specs_dir(config)
            assert result_dir is None
            assert error is not None
            assert error["data"]["error_code"] == "NOT_FOUND"

    def test_blank_string_config_falls_back(self):
        config = MagicMock(specs_dir="   ")
        with patch(
            "foundry_mcp.tools.unified.common.find_specs_directory",
            return_value=Path("/fallback"),
        ):
            result_dir, error = resolve_specs_dir(config)
            assert result_dir == Path("/fallback")
            assert error is None


# -----------------------------------------------------------------------
# 4. dispatch_with_standard_errors
# -----------------------------------------------------------------------


class TestDispatchWithStandardErrors:
    """Tests for dispatch_with_standard_errors()."""

    def _make_router(self, handler=None):
        if handler is None:
            handler = lambda **kw: {"success": True, "data": kw}
        return ActionRouter(
            tool_name="test_tool",
            actions={"do-thing": handler},
        )

    def _mock_authz_allowed(self):
        """Return patchers for authorized state."""
        return patch(
            "foundry_mcp.tools.unified.common.get_server_role",
            return_value="maintainer",
        )

    def test_success_passthrough(self):
        router = self._make_router()
        with self._mock_authz_allowed():
            result = dispatch_with_standard_errors(router, "test", "do-thing", config="cfg")
        assert result["success"] is True

    def test_unsupported_action_returns_error(self):
        router = self._make_router()
        # Unsupported actions fail before authorization check
        result = dispatch_with_standard_errors(router, "test", "bad-action")
        assert result["success"] is False
        assert "Unsupported" in result["error"]
        assert result["data"]["error_code"] == "VALIDATION_ERROR"

    def test_unsupported_action_includes_details(self):
        router = self._make_router()
        result = dispatch_with_standard_errors(
            router,
            "test",
            "bad-action",
            include_details_in_router_error=True,
        )
        assert result["success"] is False
        assert "allowed_actions" in result["data"]["details"]

    def test_generic_exception_returns_internal_error(self):
        def boom(**kw):
            raise ValueError("something broke")

        router = self._make_router(handler=boom)
        with self._mock_authz_allowed():
            result = dispatch_with_standard_errors(router, "test", "do-thing")
        assert result["success"] is False
        assert "something broke" in result["error"]
        assert result["data"]["error_code"] == "INTERNAL_ERROR"

    def test_uses_provided_request_id(self):
        router = self._make_router()
        # Unsupported actions fail before authorization check
        result = dispatch_with_standard_errors(
            router,
            "test",
            "bad-action",
            request_id="custom_id_123",
        )
        assert result["meta"]["request_id"] == "custom_id_123"

    def test_empty_exception_message_uses_class_name(self):
        def boom(**kw):
            raise RuntimeError()

        router = self._make_router(handler=boom)
        with self._mock_authz_allowed():
            result = dispatch_with_standard_errors(router, "test", "do-thing")
        assert "RuntimeError" in result["error"]

    def test_authorization_denied_for_observer_role(self):
        """Test that observer role is denied access to non-readonly actions."""
        router = self._make_router()
        with patch(
            "foundry_mcp.tools.unified.common.get_server_role",
            return_value="observer",
        ):
            result = dispatch_with_standard_errors(router, "test", "do-thing")
        assert result["success"] is False
        assert "Authorization denied" in result["error"]
        assert result["data"]["error_code"] == "AUTHORIZATION"
        assert result["data"]["details"]["role"] == "observer"

    def test_authorization_enforced_when_config_role_is_observer(self):
        """Observer config must not bypass authorization checks."""
        router = self._make_router()
        config = MagicMock()
        config.autonomy_security.role = "observer"

        with patch(
            "foundry_mcp.tools.unified.common.get_server_role",
            return_value="observer",
        ):
            result = dispatch_with_standard_errors(
                router,
                "test",
                "do-thing",
                config=config,
            )

        assert result["success"] is False
        assert result["data"]["error_code"] == "AUTHORIZATION"

    def test_authorization_denied_includes_required_role(self):
        """Test that authorization denial includes required role."""
        router = self._make_router()
        with patch(
            "foundry_mcp.tools.unified.common.get_server_role",
            return_value="observer",
        ):
            result = dispatch_with_standard_errors(router, "test", "do-thing")
        assert result["data"]["details"]["required_role"] == "maintainer"
        assert "recovery_action" in result["data"]["details"]

    def test_authorization_denied_emits_metric(self):
        """Test that authorization denial emits authz.denied metric."""
        router = self._make_router()
        with (
            patch(
                "foundry_mcp.tools.unified.common.get_server_role",
                return_value="observer",
            ),
            patch("foundry_mcp.tools.unified.common.MetricsCollector") as mock_metrics_class,
        ):
            mock_collector = MagicMock()
            mock_metrics_class.return_value = mock_collector

            dispatch_with_standard_errors(router, "test", "do-thing")

            mock_metrics_class.assert_called_once_with(prefix="authz")
            mock_collector.counter.assert_called_once_with(
                "denied",
                labels={
                    "role": "observer",
                    "tool": "test",
                    "action": "do-thing",
                    "scope": "role",
                },
            )

    def test_authorization_denial_key_uses_client_scope_when_available(self):
        """Rate-limit key should include client ID when context provides one."""
        router = self._make_router()
        mock_tracker = MagicMock()
        mock_tracker.check_rate_limit.return_value = None

        with (
            patch(
                "foundry_mcp.tools.unified.common.get_server_role",
                return_value="observer",
            ),
            patch(
                "foundry_mcp.tools.unified.common.get_client_id",
                return_value="client-123",
            ),
            patch(
                "foundry_mcp.tools.unified.common.get_rate_limit_tracker",
                return_value=mock_tracker,
            ),
        ):
            result = dispatch_with_standard_errors(router, "test", "do-thing")

        assert result["data"]["error_code"] == "AUTHORIZATION"
        mock_tracker.check_rate_limit.assert_called_once_with("test.do-thing|client:client-123")
        mock_tracker.record_denial.assert_called_once_with("test.do-thing|client:client-123")

    def test_authorization_denial_key_falls_back_to_role_scope(self):
        """Anonymous requests should rate-limit by role scope."""
        router = self._make_router()
        mock_tracker = MagicMock()
        mock_tracker.check_rate_limit.return_value = None

        with (
            patch(
                "foundry_mcp.tools.unified.common.get_server_role",
                return_value="observer",
            ),
            patch(
                "foundry_mcp.tools.unified.common.get_client_id",
                return_value="anonymous",
            ),
            patch(
                "foundry_mcp.tools.unified.common.get_rate_limit_tracker",
                return_value=mock_tracker,
            ),
        ):
            result = dispatch_with_standard_errors(router, "test", "do-thing")

        assert result["data"]["error_code"] == "AUTHORIZATION"
        mock_tracker.check_rate_limit.assert_called_once_with("test.do-thing|role:observer")
        mock_tracker.record_denial.assert_called_once_with("test.do-thing|role:observer")

    def test_untrusted_principal_hints_do_not_influence_scope(self):
        """Rate-limit scope must ignore caller-supplied principal hints."""
        router = self._make_router()
        mock_tracker = MagicMock()
        mock_tracker.check_rate_limit.return_value = None

        with (
            patch(
                "foundry_mcp.tools.unified.common.get_server_role",
                return_value="observer",
            ),
            patch(
                "foundry_mcp.tools.unified.common.get_client_id",
                return_value=None,
            ),
            patch(
                "foundry_mcp.tools.unified.common.get_rate_limit_tracker",
                return_value=mock_tracker,
            ),
        ):
            result = dispatch_with_standard_errors(
                router,
                "test",
                "do-thing",
                actor_id="attacker-supplied",
            )

        assert result["data"]["error_code"] == "AUTHORIZATION"
        mock_tracker.check_rate_limit.assert_called_once_with("test.do-thing|role:observer")
        mock_tracker.record_denial.assert_called_once_with("test.do-thing|role:observer")

    def test_authorization_allowed_for_maintainer(self):
        """Test that maintainer role has access to all actions."""
        router = self._make_router()
        with patch(
            "foundry_mcp.tools.unified.common.get_server_role",
            return_value="maintainer",
        ):
            result = dispatch_with_standard_errors(router, "test", "do-thing")
        assert result["success"] is True

    def test_authorization_check_after_action_validation(self):
        """Test that authorization check happens after action validation."""
        router = self._make_router()
        # Request for non-existent action should fail with VALIDATION_ERROR,
        # not AUTHORIZATION
        with patch(
            "foundry_mcp.tools.unified.common.get_server_role",
            return_value="observer",
        ):
            result = dispatch_with_standard_errors(router, "test", "nonexistent")
        assert result["data"]["error_code"] == "VALIDATION_ERROR"
        assert "Unsupported" in result["error"]


# -----------------------------------------------------------------------
# 5. make_validation_error_fn
# -----------------------------------------------------------------------


class TestMakeValidationErrorFn:
    """Tests for make_validation_error_fn()."""

    def test_returns_callable(self):
        fn = make_validation_error_fn("task")
        assert callable(fn)

    def test_error_message_includes_tool_name(self):
        fn = make_validation_error_fn("task")
        result = fn(
            field="spec_id",
            action="prepare",
            message="is required",
            request_id="req_123",
        )
        assert result["success"] is False
        assert "task.prepare" in result["error"]
        assert "spec_id" in result["error"]

    def test_default_remediation(self):
        fn = make_validation_error_fn("authoring")
        result = fn(
            field="name",
            action="create",
            message="cannot be empty",
            request_id="req_123",
        )
        assert "Provide a valid 'name' value" in str(result)

    def test_custom_remediation(self):
        fn = make_validation_error_fn("authoring")
        result = fn(
            field="name",
            action="create",
            message="cannot be empty",
            request_id="req_123",
            remediation="Use a non-empty string",
        )
        assert "Use a non-empty string" in str(result)

    def test_custom_error_code(self):
        fn = make_validation_error_fn("task")
        result = fn(
            field="spec_id",
            action="prepare",
            message="is required",
            request_id="req_123",
            code=ErrorCode.MISSING_REQUIRED,
        )
        assert result["data"]["error_code"] == "MISSING_REQUIRED"

    def test_auto_generates_request_id(self):
        fn = make_validation_error_fn("task", include_request_id=True)
        with (
            patch(
                "foundry_mcp.tools.unified.common.get_correlation_id",
                return_value=None,
            ),
            patch(
                "foundry_mcp.tools.unified.common.generate_correlation_id",
                return_value="task_auto123",
            ),
        ):
            result = fn(
                field="spec_id",
                action="prepare",
                message="is required",
            )
            assert result["meta"]["request_id"] == "task_auto123"

    def test_no_request_id_when_disabled(self):
        fn = make_validation_error_fn("journal", include_request_id=False)
        result = fn(
            field="spec_id",
            action="add",
            message="is required",
        )
        # Should not crash, request_id will be None
        assert result["success"] is False

    def test_details_include_field_and_action(self):
        fn = make_validation_error_fn("environment")
        result = fn(
            field="path",
            action="setup",
            message="invalid",
            request_id="req_123",
        )
        details = result["data"]["details"]
        assert details["field"] == "path"
        assert details["action"] == "environment.setup"
