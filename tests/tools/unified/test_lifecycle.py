"""Tests for unified lifecycle tool behavior.

Tests that _dispatch_lifecycle_action catches exceptions and returns error responses
instead of crashing the MCP server.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.lifecycle import MoveResult


@pytest.fixture
def mock_config():
    """Create a mock ServerConfig."""
    config = MagicMock()
    return config


class TestLifecycleDispatchExceptionHandling:
    """Tests for _dispatch_lifecycle_action exception handling."""

    def test_dispatch_catches_exceptions(self, mock_config):
        """_dispatch_lifecycle_action should catch exceptions and return error response."""
        from foundry_mcp.tools.unified.lifecycle import _dispatch_lifecycle_action

        with patch(
            "foundry_mcp.tools.unified.lifecycle._LIFECYCLE_ROUTER"
        ) as mock_router:
            mock_router.allowed_actions.return_value = ["archive"]
            mock_router.dispatch.side_effect = RuntimeError("Lifecycle transition failed")

            result = _dispatch_lifecycle_action(
                action="archive",
                payload={"spec_id": "test-spec"},
                config=mock_config,
            )

        # Should return error response, not raise exception
        assert result["success"] is False
        assert "Lifecycle transition failed" in result["error"]
        assert result["data"]["error_type"] == "internal"
        assert result["data"]["details"]["action"] == "archive"
        assert result["data"]["details"]["error_type"] == "RuntimeError"

    def test_dispatch_handles_empty_exception_message(self, mock_config):
        """_dispatch_lifecycle_action should handle exceptions with empty messages."""
        from foundry_mcp.tools.unified.lifecycle import _dispatch_lifecycle_action

        with patch(
            "foundry_mcp.tools.unified.lifecycle._LIFECYCLE_ROUTER"
        ) as mock_router:
            mock_router.allowed_actions.return_value = ["archive"]
            mock_router.dispatch.side_effect = RuntimeError()

            result = _dispatch_lifecycle_action(
                action="archive",
                payload={"spec_id": "test-spec"},
                config=mock_config,
            )

        # Should use class name when message is empty
        assert result["success"] is False
        assert "RuntimeError" in result["error"]

    def test_dispatch_logs_exception(self, mock_config, caplog):
        """_dispatch_lifecycle_action should log exceptions."""
        import logging

        from foundry_mcp.tools.unified.lifecycle import _dispatch_lifecycle_action

        with caplog.at_level(logging.ERROR):
            with patch(
                "foundry_mcp.tools.unified.lifecycle._LIFECYCLE_ROUTER"
            ) as mock_router:
                mock_router.allowed_actions.return_value = ["archive"]
                mock_router.dispatch.side_effect = ValueError("test error")

                _dispatch_lifecycle_action(
                    action="archive",
                    payload={"spec_id": "test-spec"},
                    config=mock_config,
                )

        assert "test error" in caplog.text


class TestLifecycleWriteLockWorkspaceNormalization:
    """Lifecycle path variants should normalize to workspace root for write-lock checks."""

    @staticmethod
    def _make_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
        workspace = tmp_path / "ws"
        specs_dir = workspace / "specs"
        for folder in ("active", "pending", "completed", "archived"):
            (specs_dir / folder).mkdir(parents=True, exist_ok=True)

        spec_file = specs_dir / "active" / "test-spec.json"
        spec_file.write_text("{}", encoding="utf-8")
        return workspace, specs_dir, spec_file

    @pytest.mark.parametrize(
        "path_variant",
        [
            "workspace",
            "specs_dir",
            "spec_file",
        ],
    )
    def test_move_normalizes_path_before_write_lock(self, tmp_path: Path, path_variant: str):
        from foundry_mcp.tools.unified.lifecycle import _handle_move

        workspace, specs_dir, spec_file = self._make_workspace(tmp_path)
        config = ServerConfig(specs_dir=specs_dir, log_level="WARNING")

        if path_variant == "workspace":
            path = str(workspace)
        elif path_variant == "specs_dir":
            path = str(specs_dir)
        else:
            path = str(spec_file)

        with patch(
            "foundry_mcp.tools.unified.lifecycle._check_autonomy_write_lock"
        ) as mock_lock, patch(
            "foundry_mcp.tools.unified.lifecycle.move_spec"
        ) as mock_move:
            mock_lock.return_value = None
            mock_move.return_value = MoveResult(
                success=True,
                spec_id="test-spec",
                from_folder="active",
                to_folder="archived",
                old_path=str(spec_file),
                new_path=str(specs_dir / "archived" / "test-spec.json"),
            )

            resp = _handle_move(
                config=config,
                spec_id="test-spec",
                to_folder="archived",
                path=path,
            )

        assert resp["success"] is True
        assert mock_lock.call_count == 1
        assert mock_lock.call_args.kwargs["workspace"] == str(workspace)
